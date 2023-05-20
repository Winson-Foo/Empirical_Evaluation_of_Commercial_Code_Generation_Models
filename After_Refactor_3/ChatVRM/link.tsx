import React from "react";
import { LinkProps } from "../types";

const Link: React.FC<LinkProps> = ({ url, label }) => {
  return (
    <a className="link" href={url}>
      {label}
    </a>
  );
};

export default Link;