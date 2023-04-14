%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rutgers Beamer Class File
%% Author: Kailong Wang
%% Description: This file creates a custom beamer class for the Rutgers University, the State University of New Jersey.
%% Original Template: https://www.overleaf.com/latex/templates/template-beamer-lepten-ufsc/vjdrghyjyvrp 
%% Color Profile based on the instruction from: http://identity.rutgers.edu/about-identity/colors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{formats/RU}[02/11/2023 v0.0.1 Rutgers template beamer]

% \PassOptionsToPackage{dvipsnames}{xcolor} 
\LoadClass[compress, 10pt]{beamer}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Putting the background image in the frames
\usebackgroundtemplate{
\parbox[c][\paperheight][c]{\paperwidth}{\centering\transparent{0.1}\includegraphics[height=0.8\paperheight]{../formats/RULogos/RU_SHIELD_BLACK.eps}}
}
\logo{\includegraphics[scale=0.1]{../formats/RULogos/RU_SHIELD_CMYK_RED.eps}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Packages Used

%% Font Packages
\RequirePackage[english]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{libertinus}
\RequirePackage{microtype}
\RequirePackage{lmodern}

%% Math Packages
\RequirePackage{amsfonts,amsmath,amssymb} %Math Font Packages
\RequirePackage{mathtools} %Better Math Environments
\RequirePackage{mleftright}  %Fixes some annoying spacing issues
\RequirePackage{bm} %Boldface math, load after other math font packages

%% Table Packages
\RequirePackage{tabularx} 
\RequirePackage{multirow} 
\RequirePackage{colortbl}
\RequirePackage{array}

%% Theorems etc (also  problems)
\RequirePackage[shortlabels]{enumitem} %Better List
\RequirePackage{hyperref} %Fantastic crossrefer
\RequirePackage{amsthm} %Better Theorems

%% Code Packages
\RequirePackage{algorithm2e} % Package to create pseudo-code
\RequirePackage{listings} % Package to insert code
\RequirePackage{keyval}

%% Display Packages
\RequirePackage[document]{ragged2e} % Package to justify text
\RequirePackage{csquotes} % Package to facilities quotations
\RequirePackage{multicol} % Package to use multicols

%% Other Packages
\RequirePackage{calc}
\RequirePackage{rotating}
\RequirePackage{pgf, tikz}
\RequirePackage{epstopdf}
\RequirePackage[backend=biber, style=numeric, sorting=none]{biblatex}

%% Graphics Packages
\RequirePackage{graphicx} %Better Graphics
\RequirePackage[export]{adjustbox} %Better Graphics
% \RequirePackage[dvipsnames,svgnames]{xcolor} %Must load before tcolorbox
\RequirePackage[breakable]{tcolorbox} %Customized Box
\tcbuselibrary{skins}
\usepackage{transparent}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Colors
\definecolor{black}{RGB}{0, 0, 0}
\definecolor{white}{RGB}{255, 255, 255}
\definecolor{blue}{RGB}{0, 83, 134}
\definecolor{yellow}{RGB}{206, 177, 68}
\definecolor{green}{RGB}{0, 146, 62}
\definecolor{RUgray}{RGB}{91,103,112} % gray = PANTONE 431
\definecolor{RUred}{RGB}{204,0,51} % red = PANTONE 186(250 anniversary version)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Style of the blocks 
\beamerboxesdeclarecolorscheme{blockRU}{RUred}{white}
\beamer@autopdfinfotrue
\beamer@notesnormalstrue

\usecolortheme[named=RUred]{structure}
\useinnertheme{rounded}
\useoutertheme[subsection=false]{miniframes}
\setbeamertemplate{navigation symbols}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Header and foot
\setbeamercolor{section in head/foot}{bg=RUred, fg=white}
% \setbeamercolor{subsection in head/foot}{bg=RUgray, fg=white}
% \setbeamerfont{block title}{size={}}

\setbeamercolor{author in head/foot}{bg=RUred, fg=white}
\setbeamercolor{institute in head/foot}{bg=RUred, fg=white}
\setbeamercolor{title in head/foot}{bg=RUgray, fg=white}
\setbeamercolor{date in head/foot}{bg=RUred, fg=white}

% foot style
\defbeamertemplate*{footline}{infolines theme}{
\leavevmode%
\hbox{%
\begin{beamercolorbox}[wd=.20\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
\usebeamerfont{author in head/foot}\insertshortauthor~~
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.10\paperwidth,ht=2.25ex,dp=1ex,center]{institute in head/foot}%
\usebeamerfont{institute in head/foot}\insertshortinstitute
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.40\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
\usebeamerfont{title in head/foot}\insertshorttitle
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.30\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
\usebeamerfont{date in head/foot}\insertshortdate{}\hspace*{2em}
\insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
\end{beamercolorbox}
}%
\vskip0pt%
}
\mode
<all>

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Create the blocks of the beamer template

% draw the blocks
\setbeamertemplate{blocks}[rounded][shadow=true]
% \setbeamercolor{separation line}{use=structure,bg=structure.fg!50!bg} % color on the top 

% block
\setbeamercolor*{block title}{fg=white, bg=RUred}
\setbeamercolor*{block body}{fg=black, bg=white}
% alert block
\setbeamercolor*{block title alerted}{fg=white, bg=blue}
\setbeamercolor*{block body alerted}{fg=black, bg=white}
% example block
\setbeamercolor*{block title example}{fg=white, bg=green}
\setbeamercolor*{block body example}{fg=black, bg=white}

% creating the environment the block
\AtBeginEnvironment{exampleblock}{%
  \setbeamercolor{itemize item}{fg=green!70}
}
\AtBeginEnvironment{alertblock}{%
  \setbeamercolor{itemize item}{fg=blue!70}
}
\AtBeginEnvironment{block}{%
  \setbeamercolor{itemize item}{fg=RUred!70}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Determine some color for the text
\setbeamercolor*{normal text}{fg=black,bg=white}
\setbeamercolor*{example text}{fg=yellow}
\setbeamercolor*{alerted text}{fg=RUred}
\setbeamercolor{itemize item}{fg=RUred!70}
\setbeamercolor{enumerate item}{fg=black!70}
\setbeamercolor{success}{fg=green}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% new commands
% command of the example
\renewcommand{\example}[1]{\textcolor{RUred}{\textbf{#1}}}

% command of the emphases 
\renewcommand{\emph}[1]{\textcolor{blue}{\textbf{#1}}}

% create the alert box
\newcommand{\alertbox}[1]{
\begin{flushleft}
\fcolorbox{red}{white}{
\begin{minipage}{0.9\textwidth}
#1
\end{minipage}
}
\end{flushleft}
}

% create the simple box
\newcommand{\simplebox}[1]{
\begin{flushleft}
\fcolorbox{blue}{white}{
\begin{minipage}{0.9\textwidth}
#1
\end{minipage}
}
\end{flushleft}
}

% create the success box
\newcommand{\successbox}[1]{
\begin{flushleft}
\fcolorbox{green}{white}{
\begin{minipage}{0.9\textwidth}
#1
\end{minipage}
}
\end{flushleft}
}

% create the source of the image
\newcommand{\source}[1]{
    \caption*{\textcolor{blue}{Fonte:} {#1}} 
}

% create a command for RU
\newcommand{\rutgers}{
  \bfseries
  \normalsize{Rutgers University, the State University of New Jersey}
}

% department command
\newcommand{\department}[1]{
    \vspace*{0.2cm}
    \bfseries
    \normalsize{#1}
}

% 
\newcommand{\email}[1]{
    \texttt{
      \href{mailto:#1}{#1}
    }
}

