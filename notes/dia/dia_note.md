# Note on using Dia
1. draw you dia diagram.
2. export dia as `<name>.tex` file.
3. edit `<name>.tex` file
    * add `documentclass` et. al. (see `basicworkflow.tex`, the part edited by Leo as an example)
        ```latex
        \documentclass{standalone}
        \usepackage{subfigure}
        \usepackage{tikz}
        \begin{document}
        <original contents>
        \end{document}
        ```
    * add math equations at desired position.
4. compile it use either of the following two commands
    ```bash
    $ xelatex <name>.tex
    $ pdflatex <name>.tex
    ```
    note here `pgf` is required, run `sudo apt-get install pgf` to install it.
