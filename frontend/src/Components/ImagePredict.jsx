import { useState } from "react";
import { useDispatch } from "react-redux";
import { Spinner } from "reactstrap";
import { Button, Image, Input, Label, Table } from "semantic-ui-react";
import {
  addInput,
  getData,
  uploadContent,
  deleteInput,
} from "../Store/Actioncreator";

function ImagePredict(props) {
  const { inputs } = props;
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(false);
  const handleUpload = (e) => {
    setLoading(true);
    const file = e?.target?.files[0];
    if (file) {
      const data = {
        file: file,
      };
      dispatch(uploadContent(data)).then((res) => {
        dispatch(getData({ apiUrl: inputs.url, data: { url: res } })).then((pred) => {
          dispatch(addInput({ image: res, pred: pred }));
          setLoading(false);
        });
      });
    }
  };
  const deleteRow = (i) => {
    dispatch(deleteInput({ index: i }));
  };

  return (
    <Table>
      <Table.Header>
        <Table.Row>
          <Table.HeaderCell>Image</Table.HeaderCell>
          <Table.HeaderCell>Prediction</Table.HeaderCell>
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {inputs?.images.map((image, i) => (
          <Table.Row key={i}>
            <Table.Cell>
              <Image src={image.img} size="small" />
              <Button icon="trash" onClick={() => deleteRow(i)} />
            </Table.Cell>
            <Table.Cell>
              <Label> {image.data}</Label>
            </Table.Cell>
          </Table.Row>
        ))}
      </Table.Body>
      <Table.Footer>
        <Table.Row>
          <Table.HeaderCell />
          <Table.HeaderCell colSpan="3">
            {loading ? (
              <>
                <p className="text-muted">
                  <Spinner />
                  Uploading and analyzing the image
                </p>
              </>
            ) : (
              <Input
                type="file"
                accept="Image/*"
                onChange={handleUpload}
                className="float-right"
              />
            )}
          </Table.HeaderCell>
        </Table.Row>
      </Table.Footer>
    </Table>
  );
}

export default ImagePredict;
