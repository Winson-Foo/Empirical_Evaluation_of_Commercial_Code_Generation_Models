import React, { ButtonHTMLAttributes } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  primary?: boolean;
  disabled?: boolean;
};

const TextButton = ({ primary, disabled, className, children, ...props }: Props) => {
  const baseClass = "px-24 py-8 text-white font-bold rounded-oval " + className;
  const primaryClass = "bg-primary hover:bg-primary-hover active:bg-primary-press-press";
  const disabledClass = "bg-primary-disabled";

  return (
    <button
      {...props}
      className={`${baseClass} ${primary ? primaryClass : ""} ${
        disabled ? disabledClass : ""
      }`}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default TextButton;