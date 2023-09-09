import LoginLayout from 'components/pages/login/LoginLayout';
import LoginForm from 'components/pages/login/LoginForm';

export default function LoginPage({ isLoginDisabled }) {
  // If login is disabled, return null
  if (isLoginDisabled) {
    return null;
  }

  // Otherwise, render the login form
  return (
    <LoginLayout title="Login">
      <LoginForm />
    </LoginLayout>
  );
}

export async function getServerSideProps() {
  // Get the value of DISABLE_LOGIN environment variable and convert it to boolean
  const isLoginDisabled = !!process.env.DISABLE_LOGIN;

  return {
    props: {
      isLoginDisabled,
    },
  };
}

