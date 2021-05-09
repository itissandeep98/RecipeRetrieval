import { useDispatch } from "react-redux";
import { Form, Input } from "semantic-ui-react";
import { addApiUrl } from "../Store/Actioncreator";

function AddUrl() {
  const dispatch = useDispatch();
  const onChange = (e) => {
    dispatch(addApiUrl({ url: e.target.value }));
  };

  return (
    <Form>
      <label>Add API URL</label>
      <Input fluid placeholder="API URL" onChange={onChange} />
    </Form>
  );
}

export default AddUrl;
