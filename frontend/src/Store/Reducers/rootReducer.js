import { combineReducers } from "redux";
import inputReducer from "./Inputs";

const rootReducer = combineReducers({
  input: inputReducer,
});

export default rootReducer;
