use std::net::SocketAddr;
use std::sync::Arc;

use quinn::Connection;
use rcgen::{generate_simple_self_signed, CertifiedKey};
use rustls::ClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use thiserror::Error;

use crate::crypto::identity::NodeKeypair;
use crate::network::messages::{MessageError, NetworkMessage, decode_message, encode_message};
use crate::network::messages::NodeId;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TLS error: {0}")]
    Tls(String),
    #[error("connect error: {0}")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection error: {0}")]
    ConnectionError(#[from] quinn::ConnectionError),
    #[error("send error: {0}")]
    Send(#[from] quinn::WriteError),
    #[error("recv error: {0}")]
    Recv(#[from] quinn::ReadError),
    #[error("recv exact error: {0}")]
    RecvExact(#[from] quinn::ReadExactError),
    #[error("endpoint closed")]
    Closed,
    #[error("certificate generation failed: {0}")]
    CertGen(String),
    #[error("message error: {0}")]
    Message(#[from] MessageError),
}

// ---------------------------------------------------------------------------
// SkipServerVerification — accepts any cert (P2P, no CA)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::ED25519,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
        ]
    }
}

// ---------------------------------------------------------------------------
// QuicTransport
// ---------------------------------------------------------------------------

/// QUIC transport layer backed by quinn 0.11.
///
/// Each node binds one endpoint that acts as both server and client.
/// TLS uses a self-signed rcgen certificate — no CA needed for P2P.
pub struct QuicTransport {
    pub endpoint: quinn::Endpoint,
    pub node_id: NodeId,
}

impl QuicTransport {
    /// Binds a QUIC endpoint on `addr`.
    ///
    /// Generates a self-signed TLS certificate. The client side skips
    /// certificate verification — identity is established via ed25519 signatures
    /// on the message layer instead.
    pub async fn bind(addr: SocketAddr, keypair: &NodeKeypair) -> Result<Self, TransportError> {
        // 1. Generate self-signed certificate via rcgen.
        let CertifiedKey { cert, key_pair } =
            generate_simple_self_signed(vec!["nexus".to_string()])
                .map_err(|e| TransportError::CertGen(e.to_string()))?;

        let cert_der: CertificateDer<'static> = cert.der().clone();
        let key_der: PrivateKeyDer<'static> =
            PrivatePkcs8KeyDer::from(key_pair.serialize_der()).into();

        // 2. Server config — present the self-signed cert.
        let server_config =
            quinn::ServerConfig::with_single_cert(vec![cert_der], key_der)
                .map_err(|e| TransportError::Tls(e.to_string()))?;

        // 3. Client config — skip cert verification for P2P.
        let rustls_client = ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();

        let client_config = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(rustls_client)
                .map_err(|e| TransportError::Tls(e.to_string()))?,
        ));

        // 4. Create endpoint.
        let mut endpoint = quinn::Endpoint::server(server_config, addr)?;
        endpoint.set_default_client_config(client_config);

        Ok(Self { endpoint, node_id: keypair.node_id() })
    }

    /// Opens a QUIC connection to `addr`.
    pub async fn connect(&self, addr: SocketAddr) -> Result<Connection, TransportError> {
        // "nexus" matches the SAN in our self-signed cert.
        Ok(self.endpoint.connect(addr, "nexus")?.await?)
    }

    /// Accepts the next incoming QUIC connection.
    pub async fn accept(&self) -> Result<Connection, TransportError> {
        self.endpoint
            .accept()
            .await
            .ok_or(TransportError::Closed)?
            .await
            .map_err(TransportError::ConnectionError)
    }

    /// Sends a `NetworkMessage` over `conn` using a unidirectional stream.
    ///
    /// Wire format: 4-byte big-endian length prefix + msgpack payload.
    pub async fn send(conn: &Connection, msg: &NetworkMessage) -> Result<(), TransportError> {
        let payload = encode_message(msg)?;
        let len = (payload.len() as u32).to_be_bytes();

        let mut stream = conn.open_uni().await?;
        stream.write_all(&len).await?;
        stream.write_all(&payload).await?;
        stream.finish().map_err(|_| TransportError::Closed)?;
        Ok(())
    }

    /// Receives one `NetworkMessage` from `conn`.
    pub async fn recv(conn: &Connection) -> Result<NetworkMessage, TransportError> {
        let mut stream = conn.accept_uni().await?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).await?;

        Ok(decode_message(&payload)?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::messages::{MessageType, QueryRequest};

    #[tokio::test]
    async fn send_recv_query_request() {
        // Install ring crypto provider (idempotent — ok() swallows already-set error).
        rustls::crypto::ring::default_provider().install_default().ok();

        let kp_server = NodeKeypair::generate();
        let kp_client = NodeKeypair::generate();

        let server =
            QuicTransport::bind("127.0.0.1:0".parse().unwrap(), &kp_server).await.unwrap();
        let server_addr = server.endpoint.local_addr().unwrap();

        let client =
            QuicTransport::bind("127.0.0.1:0".parse().unwrap(), &kp_client).await.unwrap();

        let qr = QueryRequest { terms: vec!["rust".to_string()], limit: 10, request_id: 99 };
        let payload = encode_message(&qr).unwrap();
        let msg = NetworkMessage {
            kind: MessageType::QueryRequest,
            payload,
            sender: kp_client.node_id(),
            signature: [0u8; 64],
        };

        let (send_res, accept_res) =
            tokio::join!(client.connect(server_addr), server.accept());

        let client_conn = send_res.unwrap();
        let server_conn = accept_res.unwrap();

        let (send_result, recv_result) = tokio::join!(
            QuicTransport::send(&client_conn, &msg),
            QuicTransport::recv(&server_conn),
        );

        send_result.unwrap();
        let received = recv_result.unwrap();

        assert_eq!(received.sender, kp_client.node_id());
        let decoded_qr: QueryRequest = decode_message(&received.payload).unwrap();
        assert_eq!(decoded_qr.request_id, 99);
        assert_eq!(decoded_qr.terms, vec!["rust".to_string()]);
    }
}
