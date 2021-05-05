import { Label } from "semantic-ui-react";

function Results(props) {
  const { inputs } = props;
  let query = "";
  inputs.ingredients.map((ing) => (query += ing + ", "));
  inputs.images.map((ing) => (query += ing.data + ", "));
  query = query.slice(0, -2);

  return (
    <div>
      Final Query: <Label>{query}</Label>
    </div>
  );
}

export default Results;
