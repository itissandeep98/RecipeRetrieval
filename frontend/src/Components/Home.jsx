import { Col, Container, Row } from "reactstrap";
import { Tab } from "semantic-ui-react";
import { connect } from "react-redux";
import ImagePredict from "./ImagePredict";
import Comparators from "./Comparators";

function Home(props) {
  const res = props.model;
  const panes = [
    {
      menuItem: "Image Prediction",
      render: () => (
        <Tab.Pane attached={false}>
          <ImagePredict res={res} />
        </Tab.Pane>
      ),
    },
    {
      menuItem: "Comparators",
      render: () => (
        <Tab.Pane attached={false}>
          <Comparators res={res.result} />
        </Tab.Pane>
      ),
    },
  ];
  return (
    <Container>
      <Row>
        <Col className="text-center shadow p-3 bg-white mt-3 rounded">
          <h1>Information Retrieval Project</h1>
          <h3>Recipe Retrieval System</h3>
        </Col>
      </Row>

      <Tab menu={{ pointing: true }} panes={panes} className="mt-5" />
    </Container>
  );
}

const mapStateToProps = (state) => ({
  model: state.model,
});

export default connect(mapStateToProps, null)(Home);
