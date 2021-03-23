import { Col, Container, Row } from "reactstrap";
import { Form, Input } from "semantic-ui-react";
import { useDispatch } from "react-redux";
import { uploadContent } from "../Store/Actioncreator";
import { useState } from "react";

function Home(props) {
  const [img, setImg] = useState("");
  const dispatch = useDispatch();
  const handleUpload = (e) => {
    const file = e?.target?.files[0];
    if (file) {
      console.log(file);
      const data = {
        file: file,
      };
      dispatch(uploadContent(data)).then((res) => setImg(res));
    }
  };
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
              <Input type="file" accept="Image/*" onChange={handleUpload} />
            </Form.Field>
          </Form>
        </Col>
      </Row>
    </Container>
  );
}

export default Home;
