import { createContext } from "react";

export function createViewerContext(viewer) {
  return createContext({ viewer });
}