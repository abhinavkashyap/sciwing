import React from 'react';
import Paper from '@material-ui/core/Paper';
import Chip from '@material-ui/core/Chip';
import './categorychiparray.css';


class CategoryChipArray extends React.Component {
    render() {
        const { chipArrayData, onChipClick } = this.props;
        return (
            <Paper className={"category-chip-array-wrapper"}>
                {
                    chipArrayData.map((chipData, arrayIdx) => {
                        return (
                            <Chip
                                key={chipData.key}
                                clickable={true}
                                label={chipData.label}
                                variant={'outlined'}
                                color={'primary'}
                                onClick={() => { onChipClick(chipData.label) }}
                            />

                        )
                    })
                }
            </Paper>
        )
    }
}

export default CategoryChipArray