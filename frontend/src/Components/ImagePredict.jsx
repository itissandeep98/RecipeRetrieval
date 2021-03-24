import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { Col, Row } from "reactstrap";
import { Form, Image, Input } from "semantic-ui-react";
import { getData, uploadContent } from "../Store/Actioncreator";

function ImagePredict(props) {
  const { res } = props;
  const dispatch = useDispatch();
  const [img, setImg] = useState("");
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
    <Row className=" shadow p-3 bg-white rounded">
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
  );
}

export default ImagePredict;
