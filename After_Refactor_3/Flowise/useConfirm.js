import { useContext } from 'react'
import ConfirmContext from 'store/context/ConfirmContext'
import { HIDE_CONFIRM, SHOW_CONFIRM } from 'store/actions'

const useConfirm = () => {
    const [confirmState, dispatch] = useContext(ConfirmContext)
    let resolveCallback

    const closeConfirm = () => {
        dispatch({ type: HIDE_CONFIRM })
    }

    const handleResolve = (value) => {
        closeConfirm()
        resolveCallback(value)
    }

    const onConfirm = () => handleResolve(true)

    const onCancel = () => handleResolve(false)

    const showConfirm = (payload) => {
        dispatch({ type: SHOW_CONFIRM, payload })
        return new Promise((resolve) => {
            resolveCallback = resolve
        })
    }

    return { confirm: showConfirm, onConfirm, onCancel, confirmState }
}

export default useConfirm

