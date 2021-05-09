import { useDispatch } from "react-redux";
import { Dropdown } from "semantic-ui-react";
import { addCommonIngredient } from "../Store/Actioncreator";

function CommonIngredients(props) {
  const { inputs } = props;
  const dispatch = useDispatch();
  const options = [
    { key: 0, text: "Pepper", value: "Pepper" },
    { key: 1, text: "Turmeric", value: "Turmeric" },
    { key: 2, text: "Cilantro", value: "Cilantro" },
    { key: 3, text: "flour", value: "flour" },
    { key: 4, text: "Salt", value: "Salt" },
    { key: 5, text: "Olive oil", value: "Olive" },
    { key: 6, text: "Cumin", value: "Cumin" },
    { key: 7, text: "Cream", value: "Cream" },
    { key: 8, text: "Cinnamon", value: "Cinnamon" },
    { key: 9, text: "Thyme", value: "Thyme" },
    { key: 10, text: "Cloves", value: "Cloves" },
    { key: 11, text: "Ketchup", value: "Ketchup" },
    { key: 12, text: "Vanilla", value: "Vanilla" },
    { key: 13, text: "Baking Powder", value: "Baking" },
    { key: 14, text: "Wine", value: "Wine" },
    { key: 15, text: "Sugar", value: "Sugar" },
    { key: 16, text: "Peanut Butter", value: "Peanut Butter" },
    { key: 17, text: "cornstarch", value: "cornstarch" },
    { key: 18, text: "Soda", value: "Soda" },
    { key: 19, text: "Mustard", value: "Mustard" },
    { key: 20, text: "Soy(Sauce)", value: "soy" },
  ];

  const modify = (e, { value }) => {
    dispatch(addCommonIngredient({ items: value }));
  };
  return (
    <>
      <h3>Select some of the common ingredients from below</h3>
      <Dropdown
        placeholder="Ingredients"
        fluid
        multiple
        search
        selection
        clearable
        options={options}
        value={inputs}
        onChange={modify}
      />
    </>
  );
}

export default CommonIngredients;
