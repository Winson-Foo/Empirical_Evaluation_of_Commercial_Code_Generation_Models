import { BellIcon, DotIcon } from "@charcoal-ui/icons";
import { ButtonHTMLAttributes } from "react";
import classNames from "classnames";
import styles from "./IconButton.module.css";
type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  iconName: keyof KnownIconType;
  isProcessing: boolean;
  label?: string;
};

export const IconButton = ({
  iconName,
  isProcessing,
  label,
  ...rest
}: Props) => {
  const icon = isProcessing ? <DotIcon scale="1" /> : <BellIcon scale="1" />;
  const buttonClasses = classNames(
    styles.Button,
    "bg-primary",
    "hover:bg-primary-hover",
    "active:bg-primary-press",
    "disabled:bg-primary-disabled",
    "text-white",
    "rounded-16",
    "text-sm",
    "p-8",
    "text-center",
    "inline-flex",
    "items-center",
    "mr-2",
    rest.className
  );
  return (
    <button {...rest} className={buttonClasses}>
      {icon}
      {label && <div className={styles.Label}>{label}</div>}
    </button>
  );
};