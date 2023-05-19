import { ButtonHTMLAttributes } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  primaryClassName?: string;
  hoverClassName?: string;
  pressClassName?: string;
  disabledClassName?: string;
};

const getClassName = (props: Props) => {
  const {
    primaryClassName = "bg-primary",
    hoverClassName = "hover:bg-primary-hover",
    pressClassName = "active:bg-primary-press-press",
    disabledClassName = "disabled:bg-primary-disabled",
    className = "",
  } = props;

  return `px-24 py-8 text-white font-bold rounded-oval ${primaryClassName} ${hoverClassName} ${pressClassName} ${disabledClassName} ${className}`;
};

export const TextButton = (props: Props) => {
  return (
    <button {...props} className={getClassName(props)}>
      {props.children}
    </button>
  );
}; 