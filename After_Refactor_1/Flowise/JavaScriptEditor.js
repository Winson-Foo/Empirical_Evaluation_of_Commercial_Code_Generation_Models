// JavaScriptEditor.js
import CodeEditor from './CodeEditor'
import 'prismjs/components/prism-clike'
import 'prismjs/components/prism-javascript'

const JavaScriptEditor = props => <CodeEditor {...props} type='javascript' />

export default JavaScriptEditor