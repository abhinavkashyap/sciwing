import React from 'react';
import Snackbar from '@material-ui/core/Snackbar';
import './parsesectsnackbar.css';

class ParseSectSnackBar extends React.Component {
    render() {
        const { open, autoHideDuration, onSnackBarClose, text } = this.props;
        console.log('snack bar open', open)
        return (
            <div>
                <Snackbar
                    open={open}
                    onClose={onSnackBarClose}
                    ContentProps={{
                        'aria-describedby': 'message-id',
                    }}
                    message={<span id="message-id">{text}</span>}
                    autoHideDuration={autoHideDuration}
                />
            </div>
        )
    }
}

export default ParseSectSnackBar;