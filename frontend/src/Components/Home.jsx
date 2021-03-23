import { Col, Container, Row } from "reactstrap";
import { Form, Input } from "semantic-ui-react";

function Home() {
  return (
    <Container>
      <Row>
        <Col className="text-center shadow p-3 bg-white mt-3 rounded">
          <h1>Information Retrieval Project</h1>
          <h3>Recipe Retrieval System</h3>
        </Col>
      </Row>
      <Row>
        <Col className="text-center shadow p-3 bg-white mt-3 rounded">
          <Form>
            <Form.Field>
              <label>Upload Image</label>
              <Input type="file" accept="Image/*" />
            </Form.Field>
          </Form>
        </Col>
      </Row>
    </Container>
  );
}

export default Home;
