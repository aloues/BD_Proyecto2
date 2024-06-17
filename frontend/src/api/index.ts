import axios from "axios";
import { env } from "../../config";

export const api = axios.create({
  baseURL: env.api_url,
});

console.log("API URL", env.api_url);