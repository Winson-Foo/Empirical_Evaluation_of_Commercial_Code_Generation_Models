import { KnownIconType } from "@charcoal-ui/icons";
import { ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement>;

const Button = ({ className, ...rest }: ButtonProps) => (
  <button
    {...rest}
    className={`bg-primary hover:bg-primary-hover active:bg-primary-press disabled:bg-primary-disabled text-white rounded-16 text-sm p-8 text-center inline-flex items-center mr-2
        ${className}`}
  />
);

type IconProps = {
  name: keyof KnownIconType;
  isProcessing: boolean;
};

const Icon = ({ name, isProcessing }: IconProps) =>
  isProcessing ? (
    <pixiv-icon name={"24/Dot"} scale="1"></pixiv-icon>
  ) : (
    <pixiv-icon name={name} scale="1"></pixiv-icon>
  );

type Props = ButtonProps & {
  iconName: keyof KnownIconType;
  isProcessing: boolean;
  label?: string;
};

export const IconButton = ({ iconName, isProcessing, label, ...rest }: Props) => (
  <Button className={rest.className}>
    <Icon name={iconName} isProcessing={isProcessing} />
    {label && <div className="mx-4 font-M_PLUS_2 font-bold">{label}</div>}
  </Button>
);