import { useState } from "react";
import { useDispatch } from "react-redux";
import { Col, Row } from "reactstrap";
import {
  Accordion,
  Button,
  Form,
  Icon,
  Input,
  Label,
  Table,
} from "semantic-ui-react";
import { getResult } from "../Store/Actioncreator";

function Comparators(props) {
  const { res } = props;
  const [sentence, setSentence] = useState("");
  const [operands, setOperands] = useState("");
  const dispatch = useDispatch();
  const [activeIndex, setactiveIndex] = useState(1);
  const handleSubmit = (e) => {
    e.preventDefault();
    const data = {
      sentence,
      operands,
    };
    dispatch(getResult(data));
  };
  const handleClick = () => {
    setactiveIndex(1 - activeIndex);
  };
  return (
    <Row className=" shadow p-3 bg-white  rounded">
      <Col>
        <Form>
          <Form.Field>
            <label>Sentence</label>
            <Input
              value={sentence}
              onChange={(e) => setSentence(e.target.value)}
            />
          </Form.Field>
          <Form.Field>
            <label>Operands</label>
            <Input
              value={operands}
              onChange={(e) => setOperands(e.target.value)}
            />
          </Form.Field>
          <Button onClick={handleSubmit}>Get Result</Button>
        </Form>
      </Col>
      {res && (
        <Col xs={12} className="mt-4">
          <hr />
          <h2>Result</h2>
          <Row>
            <Col>
              <Label>
                <p className="d-inline text-info">{res?.response?.length}</p>{" "}
                files match your Query
              </Label>
              <br />
              <br />
              <Label>
                It took <p className="d-inline text-info">{res?.comparison}</p>{" "}
                Comparisons
              </Label>
            </Col>
            <Col>
              <Accordion>
                <Accordion.Title
                  active={activeIndex === 0}
                  index={0}
                  onClick={handleClick}
                >
                  <Icon name="dropdown" />
                  List of Files
                </Accordion.Title>
                <Accordion.Content active={activeIndex === 0}>
                  <Table>
                    <Table.Header>
                      <Table.Row>
                        <Table.HeaderCell>File ID</Table.HeaderCell>
                        <Table.HeaderCell>File Location</Table.HeaderCell>
                      </Table.Row>
                    </Table.Header>
                    <Table.Body>
                      {res.response.map((file) => (
                        <Table.Row>
                          <Table.Cell>{file[0]}</Table.Cell>
                          <Table.Cell>{file[1]}</Table.Cell>
                        </Table.Row>
                      ))}
                    </Table.Body>
                  </Table>
                </Accordion.Content>
              </Accordion>
            </Col>
          </Row>
        </Col>
      )}
    </Row>
  );
}

export default Comparators;
