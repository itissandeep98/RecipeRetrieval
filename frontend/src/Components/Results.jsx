import { Accordion, Button, Label } from "semantic-ui-react";
import { useDispatch } from "react-redux";
import { getResult } from "../Store/Actioncreator";
import { useState } from "react";
import { Spinner } from "reactstrap";

function Results(props) {
  const { inputs } = props;
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState([]);
  let query = "";
  inputs.ingredients.map((ing) => (query += ing + ", "));
  inputs.images.map((ing) => (query += ing.data + ", "));
  query = query.slice(0, -2);
  const dispatch = useDispatch();
  const submit = () => {
    setLoading(true);
    dispatch(getResult({ apiUrl: inputs.url, data: { query } })).then((res) => {
      setData(res);
      setLoading(false);
    });
  };

  return (
    <div>
      Final Query: <Label>{query}</Label>
      <Button disabled={loading} floated="right" onClick={submit}>
        {loading ? <Spinner /> : "Get Recipes"}
      </Button>
      <hr />
      <Accordion defaultActiveIndex={1} panels={data} />
    </div>
  );
}

export default Results;
