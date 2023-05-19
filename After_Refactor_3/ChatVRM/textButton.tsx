import { ButtonHTMLAttributes } from "react";

interface TextButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  buttonText: string;
}

const textButtonStyles = {
  default: "px-24 py-8 text-white font-bold bg-primary",
  hover: "hover:bg-primary-hover",
  active: "active:bg-primary-press-press",
  disabled: "disabled:bg-primary-disabled",
  rounded: "rounded-oval",
};

export const TextButton = ({
  buttonText,
  ...rest
}: TextButtonProps): JSX.Element => {
  const { default: defaultStyle, ...otherStyles } = textButtonStyles;

  return (
    <button
      {...rest}
      className={`${defaultStyle} ${Object.values(otherStyles).join(" ")} ${
        rest.className
      }`}
    >
      {buttonText}
    </button>
  );
};