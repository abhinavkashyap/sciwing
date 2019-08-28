import React from 'react';
import TextField from '@material-ui/core/TextField';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import Chip from '@material-ui/core/Chip';

import { uniq } from 'lodash'
import './data_vis.css';

import ParseSectSnackBar from '../snackbar/ParseSectSnackbar';
import CategoryChipArray from '../categorychiparray/CategoryChipArray';

class DataVis extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            'paper_num': 1,
            'snackbar_open': false,
            'paper_info': [],
            'categories': ['address', 'affiliation', 'author', 'bodyText',
                'category', 'construct', 'copyright', 'email', 'equation',
                'figure', 'figureCaption', 'footnote', 'keyword', 'listItem',
                'note', 'page', 'reference', 'sectionHeader', 'subsectionHeader',
                'subsubsectionHeader', 'tableCaption', 'table', 'title'
            ]
        };
        this.datavis_url = process.env.REACT_APP_DATA_EXPLORATION_URL;

        this.componentDidMount = this.componentDidMount.bind(this);
        this.fetchLines = this.fetchLines.bind(this);
        this.handleChipClick = this.handleChipClick.bind(this);

        this.handleSnackBarClose = this.handleSnackBarClose.bind(this);

    }

    handlePaperNumberChange = paper_num => event => {
        this.setState({ [paper_num]: event.target.value });
        let paper_number_entered = event.target.value;
        if (paper_number_entered <= 0 || paper_number_entered > 40) {
            this.setState({ ...this.state, 'snackbar_open': true })
        } else {
            this.fetchLines(paper_number_entered)
        }

    };

    handleSnackBarClose() {
        this.setState({ ...this.state, 'snackbar_open': false })
    }

    handleChipClick(tag_clicked) {
        console.log('get data for', tag_clicked)
        this.fetchLines(this.state.paper_num, tag_clicked)
    }

    render() {
        const { paper_num, snackbar_open, paper_info, categories } = this.state;

        let paperLines = paper_info.map((paper, paper_idx) => {
            return <TableRow key={paper['line_count']}>
                <TableCell>
                    <span >Text: </span> {paper['text']} <br />
                    <span>Tokenized Text: </span>[{paper['tokenized_text'].join(',')}]
                </TableCell>
                <TableCell>
                    <Chip
                        label={paper['label']}
                        clickable={true}
                    >
                    </Chip>
                </TableCell>
            </TableRow>
        })

        let line_categories = categories.map((category, idx) => {
            return { 'key': idx, 'label': category }
        })

        return (
            <div>
                <form noValidate autoComplete="off">
                    <TextField
                        id="standard-name"
                        label="Research Paper #"
                        placeholder="Enter numbers from 1-40"
                        value={paper_num}
                        onChange={this.handlePaperNumberChange('paper_num')}
                        margin="normal"
                    />
                </form>

                <ParseSectSnackBar
                    open={snackbar_open}
                    autoHideDuration={1000}
                    onSnackBarClose={this.handleSnackBarClose}
                    text={"Please enter a # from 1-40"}
                />
                <CategoryChipArray
                    chipArrayData={line_categories}
                    onChipClick={this.handleChipClick}
                />

                <Paper id='lines-table-container'>
                    <Table>
                        <TableBody>
                            {paperLines}
                        </TableBody>
                    </Table>
                </Paper>
            </div>

        )
    } //end of the return function

    componentDidMount() {
        this
            .fetchLines(this.state.paper_num)

    }

    fetchLines(paper_num, label) {
        let fetchUrl = this.datavis_url + "?file_no=" + paper_num
        if (label) {
            fetchUrl = fetchUrl + "&label=" + label
        }
        fetch(fetchUrl)
            .then(response => response.json())
            .then(data => {
                console.log(data)
                this.setState({ ...this.state, 'paper_info': data })
            })
    }
}

export default DataVis 