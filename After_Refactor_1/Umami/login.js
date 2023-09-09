import LoginLayout from 'components/pages/login/LoginLayout';
import LoginForm from 'components/pages/login/LoginForm';

export async function getServerSideProps() {
  const { DISABLE_LOGIN } = process.env;
  const disabled = !!DISABLE_LOGIN;
  return { props: { disabled } };
}

export function LoginPage({ disabled }) {
  return disabled ? null : (
    <LoginLayout title="login">
      <LoginForm />
    </LoginLayout>
  );
}