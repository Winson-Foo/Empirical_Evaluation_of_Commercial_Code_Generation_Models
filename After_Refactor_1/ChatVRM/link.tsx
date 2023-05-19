type LinkProps = {
  url: string;
  label: string;
  target?: "_blank" | "_self" | "_parent" | "_top";
  rel?: "noopener noreferrer" | "nofollow";
};

const LinkClassName = "text-primary hover:text-primary-hover";

export const Link = ({ url, label, target = "_blank", rel = "noopener noreferrer" }: LinkProps) => {
  return (
    <a
      className={LinkClassName}
      target={target}
      rel={rel}
      href={url}
    >
      {label}
    </a>
  );
}; 