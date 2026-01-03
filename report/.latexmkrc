# latexmk configuration file for compiling LaTeX documents
# This file configures latexmk to use pdflatex with bibtex

# Use pdflatex for compilation
$pdf_mode = 1;  # Generate PDF using pdflatex

# Use bibtex for bibliography
$bibtex_use = 2;  # Run bibtex when needed

# Set the PDF viewer - using zathura (auto-reloads on file change!)
$pdf_previewer = 'zathura';
# Other options:
# Linux with evince: $pdf_previewer = 'evince';
# Linux with okular: $pdf_previewer = 'okular';
# macOS: $pdf_previewer = 'open -a Preview';

# Continuous preview mode settings
$preview_continuous_mode = 1;  # Enable continuous preview
$pdf_update_method = 4;  # Update PDF viewer automatically

# Clean up auxiliary files
$clean_ext = 'bbl nav snm vrb synctex.gz run.xml';

# Additional pdflatex options
$pdflatex = 'pdflatex -interaction=nonstopmode -shell-escape %O %S';

# Enable shell escape for packages like minted (syntax highlighting)
# Remove -shell-escape if you don't need it for security reasons

# Maximum number of compilation runs
$max_repeat = 12;

# Output directory - keeps source directory clean!
$out_dir = 'build';

# Ensure the output directory exists
system("mkdir -p build");

# Auxiliary directory (same as output for simplicity)
$aux_dir = 'build';

# Ensure proper handling of bibliography
$biber = 'biber %O %S';

# Show CPU time used
$show_time = 1;

# Warnings
$warnings_as_errors = 0;  # Don't treat warnings as errors

# File extensions to clean
@generated_exts = qw(aux bbl blg fdb_latexmk fls log out toc nav snm vrb synctex.gz run.xml);
