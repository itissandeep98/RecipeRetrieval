import { Col, Container, Row } from "reactstrap";
import { connect } from "react-redux";
import ImagePredict from "./ImagePredict";
import { Button, Icon } from "semantic-ui-react";
import { useState } from "react";
import CommonIngredients from "./CommonIngredients";
import Results from "./Results";

function Home(props) {
  const [index, setIndex] = useState(0);

  const steps = [
    <ImagePredict inputs={props.input.images} />,
    <CommonIngredients inputs={props.input.ingredients} />,
    <Results inputs={props.input} />,
  ];
  return (
    <Container>
      <Row>
        <Col className="text-center shadow p-3 bg-white mt-3 rounded">
          <h1 className="text-grad">Information Retrieval Project</h1>
          <h2>Recipe Retrieval System</h2>
        </Col>
      </Row>
      <Row className=" shadow p-3 mt-3 bg-white rounded">
        <Col>
          {index > 0 && (
            <Button floated="left" onClick={() => setIndex(index - 1)}>
              <Icon name="arrow alternate circle left" /> Back
            </Button>
          )}

          {index < steps.length - 1 && (
            <Button floated="right" onClick={() => setIndex(index + 1)}>
              Next <Icon name="arrow alternate circle right" />
            </Button>
          )}
          <br />
          <br />
          {steps[index]}
        </Col>
      </Row>
    </Container>
  );
}

const mapStateToProps = (state) => ({
  input: state.input,
});

export default connect(mapStateToProps, null)(Home);
