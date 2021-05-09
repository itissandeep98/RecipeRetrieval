import * as ActionTypes from "../ActionTypes";

const initState = { images: [], ingredients: [] };

const inputReducer = (state = initState, action) => {
  switch (action.type) {
    case ActionTypes.IMAGE_ADD_SUCCESS:
      let temp = {};
      temp["img"] = action.image;
      temp["data"] = action.pred;
      return { ...state, images: [...state.images, temp] };

    case ActionTypes.IMAGE_DELETE_SUCCESS:
      return {
        ...state,
        images: [
          ...state.images.slice(0, action.index),
          ...state.images.slice(action.index + 1),
        ],
      };
    case ActionTypes.INGREDIENT_ADD_SUCCESS:
      return { ...state, ingredients: [...action.items] };

    case ActionTypes.URL_ADD_SUCCESS:
      return { ...state, url: action.url };

    default:
      return state;
  }
};

export default inputReducer;
