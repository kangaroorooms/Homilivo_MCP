import { ChatWindow } from "./ChatWindow";
import "./styles.css";

const backendUrl = import.meta.env.VITE_BACKEND_URL as string | undefined;
const token = import.meta.env.VITE_DEMO_TOKEN as string | undefined;

function App() {
  if (!backendUrl || !token) {
    return (
      <div className="config-error">
        <h1>Missing configuration</h1>
        <p>
          Set <code>VITE_BACKEND_URL</code> and <code>VITE_DEMO_TOKEN</code>{" "}
          in <code>frontend-demo/.env</code> (see{" "}
          <code>frontend-demo/.env.example</code>) and restart the dev
          server.
        </p>
      </div>
    );
  }

  return <ChatWindow backendUrl={backendUrl} token={token} />;
}

export default App;
