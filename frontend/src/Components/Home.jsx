import { Col, Container, Row } from "reactstrap";
import { Form, Image, Input } from "semantic-ui-react";
import { connect, useDispatch } from "react-redux";
import { getData, uploadContent } from "../Store/Actioncreator";
import { useEffect, useState } from "react";

function Home(props) {
  const res = props.model;
  const [img, setImg] = useState("");
  const dispatch = useDispatch();
  const handleUpload = (e) => {
    const file = e?.target?.files[0];
    if (file) {
      const data = {
        file: file,
      };
      dispatch(uploadContent(data)).then((res) => setImg(res));
    }
  };

  useEffect(() => {
    if (img) {
      dispatch(getData({ url: img }));
    }
  }, [img, dispatch]);
  return (
    <Container>
      <Row>
        <Col className="text-center shadow p-3 bg-white mt-3 rounded">
          <h1>Information Retrieval Project</h1>
          <h3>Recipe Retrieval System</h3>
        </Col>
      </Row>
      <Row className=" shadow p-3 bg-white mt-3 rounded">
        <Col>
          <Form>
            <Form.Field>
              <label>Upload Image</label>
              <Input type="file" accept="Image/*" onChange={handleUpload} />
            </Form.Field>
            <Form.Field>
              <label>Image URL</label>
              <Input value={img} />
            </Form.Field>
          </Form>
        </Col>
        {img && (
          <Col>
            <Image src={img} alt="content" />
          </Col>
        )}
        <Col xs={12}>
          <h3>Response: </h3>
          <p className="text-muted">{res?.data}</p>
        </Col>
      </Row>
    </Container>
  );
}

const mapStateToProps = (state) => ({
  model: state.model,
});

export default connect(mapStateToProps, null)(Home);
