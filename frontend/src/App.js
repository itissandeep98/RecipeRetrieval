import Routing from "./Components/Routing";
import { BrowserRouter } from "react-router-dom";
import "./App.css";

function App() {
  return (
    <BrowserRouter basename={process.env.PUBLIC_URL}>
      <div className="App bg-gradient" style={{minHeight:"100vh"}}>
        <Routing />
      </div>
    </BrowserRouter>
  );
}

export default App;
