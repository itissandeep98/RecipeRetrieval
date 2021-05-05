import { Col, Container, Row } from "reactstrap";
import { connect } from "react-redux";
import ImagePredict from "./ImagePredict";

function Home(props) {
  const res = props.model;

  return (
    <Container>
      <Row>
        <Col className="text-center shadow p-3 bg-white mt-3 rounded">
          <h1 className="text-grad">Information Retrieval Project</h1>
          <h2>Recipe Retrieval System</h2>
        </Col>
      </Row>
      <ImagePredict res={res} />
    </Container>
  );
}

const mapStateToProps = (state) => ({
  model: state.model,
});

export default connect(mapStateToProps, null)(Home);
