import { combineReducers } from "redux";
import modelReducer from "./model";

const rootReducer = combineReducers({
  model: modelReducer,
});

export default rootReducer;
