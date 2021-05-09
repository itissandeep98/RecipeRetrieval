import * as ActionTypes from "./ActionTypes";
import { storage } from "../Config/fire";
import axios from "axios";

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

export const getData = ({ apiUrl, data }) => {
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
        console.log(apiUrl, error);
        dispatch({
          type: ActionTypes.DATA_FAILED,
          errmess: "Error in connection with Server",
        });
      });
  };
};

export const getResult = ({ apiUrl, data }) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.RESULT_REQUEST, data });
    return await axios
      .post(`${apiUrl}results`, data)
      .then((response) => {
        dispatch({
          type: ActionTypes.RESULT_SUCCESS,
          data: response.data,
        });
        return response?.data?.data;
      })
      .catch((error) => {
        console.log(error);
        dispatch({
          type: ActionTypes.RESULT_FAILED,
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

export const addApiUrl = (data) => {
  return async (dispatch) => {
    dispatch({ type: ActionTypes.URL_ADD_SUCCESS, ...data });
  };
};
