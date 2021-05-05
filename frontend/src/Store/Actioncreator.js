import * as ActionTypes from "./ActionTypes";
import { storage } from "../Config/fire";
import axios from "axios";
import { apiUrl } from "./Url";

export const uploadContent = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.UPLOAD_REQUEST });
    const uploadTask = storage.ref(`/${data.file.name}`).put(data.file);
    uploadTask.on(
      "state_changed",
      (snapShot) => {},
      (err) => {
        console.log(err);
        dispatch({ type: ActionTypes.UPLOAD_FAILED, errmess: err });
      },
      () => {
        dispatch({ type: ActionTypes.UPLOAD_SUCCESS });
      }
    );
    return await uploadTask.then((res) =>
      storage.ref(data.content).child(data.file.name).getDownloadURL()
    );
  };
};

export const getData = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.DATA_REQUEST });
    return await axios
      .post(`${apiUrl}`, data)
      .then((response) => {
        dispatch({
          type: ActionTypes.DATA_SUCCESS,
          data: response.data.response,
        });
        return response.data.response;
      })
      .catch((error) => {
        console.log(error);
        dispatch({
          type: ActionTypes.DATA_FAILED,
          errmess: "Error in connection with Server",
        });
      });
  };
};

export const getResult = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.COMPARATOR_REQUEST });
    return await axios
      .post(`${apiUrl}comparator`, data)
      .then((response) => {
        dispatch({
          type: ActionTypes.COMPARATOR_SUCCESS,
          data: response.data,
        });
      })
      .catch((error) => {
        console.log(error);
        dispatch({
          type: ActionTypes.COMPARATOR_FAILED,
          errmess: "Error in connection with Server",
        });
      });
  };
};

export const addInput = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.IMAGE_ADD_SUCCESS, ...data });
  };
};

export const deleteInput = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.IMAGE_DELETE_SUCCESS, ...data });
  };
};

export const addCommonIngredient = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.INGREDIENT_ADD_SUCCESS, ...data });
  };
};
