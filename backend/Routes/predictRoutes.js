import express from "express";
import prediction from "../Controllers/predictRoutesControllers.js";
const Adminrouter = express.Router();
Adminrouter.post("/predict",prediction);
export default Adminrouter;