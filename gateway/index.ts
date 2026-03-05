import { app } from "./src/api"
import { registry } from "./src/services/NodeRegistry"

setInterval(() => registry.checkHeartbeats(), 5_000)

app.listen(3000, () => {
  console.log("Nexus gateway running on http://localhost:3000")
})
