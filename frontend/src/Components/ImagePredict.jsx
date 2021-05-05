import { useState } from "react";
import { useDispatch } from "react-redux";
import { Col, Row } from "reactstrap";
import { Button, Icon, Image, Input, Table } from "semantic-ui-react";
import { getData, uploadContent } from "../Store/Actioncreator";

function ImagePredict(props) {
  const [images, setImages] = useState([]);
  const dispatch = useDispatch();
  const handleUpload = (e, i) => {
    const file = e?.target?.files[0];
    if (file) {
      const data = {
        file: file,
      };
      dispatch(uploadContent(data)).then((res) => {
        dispatch(getData({ url: res })).then((pred) => {
          console.log(pred);
          let temp = images[i] ?? {};
          temp["img"] = res;
          temp["data"] = pred;
          setImages([...images.slice(0, i), temp, ...images.slice(i + 1)]);
        });
      });
    }
  };

  const addRow = () => {
    setImages([...images, {}]);
  };

  return (
    <Row className=" shadow p-3 mt-3 bg-white rounded">
      <Col>
        <Table>
          <Table.Header>
            <Table.Row>
              <Table.HeaderCell>Upload Image</Table.HeaderCell>
              <Table.HeaderCell>Image</Table.HeaderCell>
              <Table.HeaderCell>Prediction</Table.HeaderCell>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {images.map((image, i) => (
              <Table.Row key={i}>
                <Table.Cell>
                  <Input
                    type="file"
                    accept="Image/*"
                    onChange={(e) => handleUpload(e, i)}
                  />
                </Table.Cell>
                <Table.Cell>
                  <Image src={image.img} size="small" />
                </Table.Cell>
                <Table.Cell>{image.data}</Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
          <Table.Footer>
            <Table.Row>
              <Table.HeaderCell />
              <Table.HeaderCell colSpan="4">
                <Button
                  floated="right"
                  icon
                  labelPosition="left"
                  primary
                  size="small"
                  onClick={addRow}
                >
                  <Icon name="image" /> Add Image
                </Button>
              </Table.HeaderCell>
            </Table.Row>
          </Table.Footer>
        </Table>
        <Button>Next</Button>
      </Col>
    </Row>
  );
}

export default ImagePredict;
