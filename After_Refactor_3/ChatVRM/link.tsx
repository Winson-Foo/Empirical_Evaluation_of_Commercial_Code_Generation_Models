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

export interface LinkProps {
  url: string;
  label: string;
}

.link {
  color: #007aff;
}

.link:hover {
  color: #0057d0;
}

.link:visited {
  color: #007aff;
}

import Link from "./Link";
import { LinkProps } from "../types";

const MyComponent: React.FC = () => {
  const linkData: LinkProps = { url: "https://example.com", label: "Example" };

  return (
    <div>
      <p>This is a link to an example website:</p>
      <Link {...linkData} />
    </div>
  );
};

export default MyComponent;
