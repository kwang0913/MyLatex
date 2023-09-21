%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rutgers Beamer Class File
%% Author: Kailong Wang
%% Description: This file creates a custom beamer class for the Rutgers University, the State University of New Jersey.
%% Original Template: https://www.overleaf.com/latex/templates/template-beamer-lepten-ufsc/vjdrghyjyvrp
%% Color Profile based on the instruction from: http://identity.rutgers.edu/about-identity/colors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{../Formats/RU}[2023/02/11 v0.0.1 Rutgers template beamer]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
\LoadClass[
    compress,
    xcolor={
        % dvipsnames,
        svgnames,
        },
    notheorems,
    ]{beamer}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Packages Used

%% Font Packages
\RequirePackage[english]{babel} %Multilangual support
% \RequirePackage[utf8]{inputenc} %utf8 support. beamer has this by default
\RequirePackage[T1]{fontenc} %Better Font Encoding for printing
\RequirePackage{libertinus} %Popular Math Font
\RequirePackage{microtype} %Better Font Spacing.
\RequirePackage{lmodern} %Popular Font

%% Math Packages
\RequirePackage{amsfonts,amsmath,amssymb} %Math Fonts and Symbols
\RequirePackage{mathtools} %Define math symbols
\RequirePackage{mleftright}  %Fixes some annoying spacing issues
\RequirePackage{bm} %Boldface math, load after other math font packages

%% Graphics Packages
\RequirePackage{graphicx} %Graphics backend with key-value arguments
\RequirePackage[export]{adjustbox} %More arguments for includegraphics
% \RequirePackage[svgnames]{xcolor} %Beamer has this by default
\RequirePackage[most]{tcolorbox} %Customized Box
\RequirePackage{transparent} %Transparent Images
\RequirePackage{subcaption} %Package to use subfigure

%% Table Packages
\RequirePackage{tabularx} %Including tabular with more space option
\RequirePackage{multirow} %Package to use multirow in tables
\RequirePackage{colortbl} %Package to use color in tables
\RequirePackage{array} %Math and more spacing support in tables

%% Theorems etc (also  problems)
\RequirePackage[shortlabels]{enumitem} %Better List
% \RequirePackage{hyperref} %Beamer has this by default
% \RequirePackage{cleveref} %Must load before amsthm. Conflict With Beamer
\RequirePackage{amsthm, thmtools, thm-restate} %Better Theorems

%% Code Packages
\RequirePackage{listings} %Package to insert code. Better than minted
\RequirePackage{algorithm} %Package to insert algorithm
\RequirePackage{algpseudocode} %Package to insert algorithm
% \RequirePackage{keyval}

%% Display Packages
% \RequirePackage[document]{ragged2e} %Replaced by microtype
\RequirePackage{csquotes} %Package to facilities quotations
\RequirePackage{multicol} %Package to use multicols

%% Other Packages
\RequirePackage{xspace} %Package to add space after macros
\RequirePackage{calc} %Package to calculate in Latex
\RequirePackage{rotating} %Package to rotate objects
% \RequirePackage{pgf, tikz} %Package to draw figures.
\RequirePackage{epstopdf} %Package to convert eps to pdf
\RequirePackage[backend=biber, style=verbose, sorting=none]{biblatex}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Colors
\definecolor{RUgray}{HTML}{6A757C} % gray = PANTONE 431
\definecolor{RUred}{HTML}{cc0033} % red = PANTONE 186(250 anniversary version)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Putting the background image in the frames
\usebackgroundtemplate{
\parbox[c][\paperheight][c]{\paperwidth}{\centering\transparent{0.1}\includegraphics[height=0.8\paperheight]{RULogos/RU_SHIELD_BLACK.eps}}
}

% \logo{\includegraphics[scale=0.075]{../formats/RULogos/RU_SHIELD_CMYK_RED.eps}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Style of the blocks
\usecolortheme[named=RUred]{structure}
\usefonttheme[onlymath]{serif}
\useinnertheme{circles}
\useoutertheme[subsection=false]{miniframes}
% \setbeamertemplate{navigation symbols}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Header and foot
\setbeamercolor*{section in head/foot}{fg=RUred}
% \setbeamercolor{subsection in head/foot}{bg=RUgray, fg=white}
% \setbeamerfont{block title}{size={}}

\setbeamercolor*{author in head/foot}{bg=RUred, fg=white}
\setbeamercolor*{institute in head/foot}{bg=RUred, fg=white}
\setbeamercolor*{title in head/foot}{bg=RUgray, fg=white}
\setbeamercolor*{date in head/foot}{bg=RUred, fg=white}

% Headline style
\setbeamertemplate{headline}{%
\begin{minipage}[c][0.09\paperheight]{\paperwidth}
    \begin{minipage}{0.25\paperwidth}
        ~\includegraphics[height=0.081\paperheight]{RULogos/RU_SHIELD_CMYK_RED.eps}
    \end{minipage}
    \begin{minipage}{0.75\paperwidth}
        \insertnavigation{0.75\paperwidth}
    \end{minipage}
\end{minipage}
\color{RUred}\rule{\paperwidth}{1pt}
}

% footline style
\defbeamertemplate*{footline}{infolines theme}{%
\leavevmode%
\hbox{%
\begin{beamercolorbox}[wd=.15\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
\usebeamerfont{author in head/foot}\insertshortauthor
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.15\paperwidth,ht=2.25ex,dp=1ex,center]{institute in head/foot}%
\usebeamerfont{institute in head/foot}\insertshortinstitute
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.45\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
\usebeamerfont{title in head/foot}\insertshorttitle
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.25\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
\usebeamerfont{date in head/foot}\insertshortdate{}\hspace*{4ex}
\insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
\end{beamercolorbox}
}%
}
\mode
<all>

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create the blocks of the beamer template

% draw the blocks
\setbeamertemplate{blocks}[rounded][shadow=true]

% block
\setbeamercolor*{block title}{fg=white, bg=MediumVioletRed}
\setbeamercolor*{block body}{fg=black, bg=white}
% alert block
\setbeamercolor*{block title alerted}{fg=white, bg=MediumSlateBlue}
\setbeamercolor*{block body alerted}{fg=black, bg=white}
% example block
\setbeamercolor*{block title example}{fg=white, bg=MediumSeaGreen}
\setbeamercolor*{block body example}{fg=black, bg=white}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Theorem Style
% \declaretheoremstyle[
%     spaceabove=6pt, spacebelow=6pt,
%     headfont=\normalfont\bfseries,
%     notefont=\mdseries, notebraces={(}{)},
%     bodyfont=\normalfont,
%     postheadspace=1em
%     ]{mystyle}