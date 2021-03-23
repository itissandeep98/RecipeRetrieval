import * as ActionTypes from "../ActionTypes";

const initState = {
  isLoading: false,
};

const modelReducer = (state = initState, action) => {
  switch (action.type) {
    case ActionTypes.DATA_REQUEST:
      return { ...state, errmess: null, isLoading: true };

    case ActionTypes.DATA_SUCCESS:
      return {
        ...state,
        errmess: null,
        data: action.data,
      };

    case ActionTypes.DATA_FAILED:
      return { ...state, errmess: action.errmess, isLoading: false };

    default:
      return state;
  }
};

export default modelReducer;
