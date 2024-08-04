# Crosslingo TODO

<em>[TODO.md spec & Kanban Board](https://bit.ly/3fCwKfM)</em>

### Todo

- [ ] Create website template  
- [ ] Create Python API for website  
- [ ] Find larger sources for word frequencies  
- [ ] Improve crossword generation time  
  - [ ] Pick word first instead of position  
- [ ] Find ways to reduce initial dictionary  
  - [ ] Add CEFR tag to words  
  - [ ] Find large pre-trained word2vec models  

### In Progress

- [ ] [BUG] Only pick words that have clues for the mode selected  

### Done ✓

- [x] Gather word data (definitions, frequencies, translations)  
- [x] Create word index for easy access to basic data  
- [x] Implement crossword grid logic  
- [x] Implement crossword generation  
- [x] Improve crossword generator time  
  - [x] Use standard loops instead of pandas' apply❌  
  - [x] Create templates that are then filled❌  
- [x] Find ways to reduce initial dictionary  
  - [x] Use word2vec to select a word theme  
- [x] Move index to SQLite to make updates easier  

