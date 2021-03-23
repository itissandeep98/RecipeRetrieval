import { Spinner } from "reactstrap";

function Loading() {
  return (
    <Spinner size="lg" style={{ position: "fixed", left: "50%", top: "50%" }} />
  );
}

export default Loading;
