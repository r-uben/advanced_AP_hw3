classdef tabConstructor
    methods
        function tab = open_it(obj, name)
            tab = fopen(strrep(strjoin(["secs/tab/", name, ".tex"]), " ", ""), 'w+');
        end
        function open_table_env(obj, tab)
            fprintf(tab, '%s\n', '\begin{table}[h!] \small');
        end
        function captionsetup(obj, tab, width)
            fprintf(tab, '%s%1.1f%s\n', '\captionsetup{font=small, width=', width, '\textwidth}');
        end
        function captioning(obj,tab,caption)
            fprintf(tab, '%s%s%s\n', '\caption{', caption, '} \label{tab:tab1} \vspace{0.25cm}');
        end
        function centering(obj,tab);
           fprintf(tab, '%s\n', '\centering');
        end
        function row_sep(obj,tab,sep)
            fprintf(tab, '%s%1.1f%s\n', '\def\arraystretch{', sep, '}');
        end

        function open_tabular_env(obj,tab, N)
            fprintf(tab, '%s%s%s\n', '\begin{tabular}{l|', repelem('c', N), '}' );
%             fprintf(tab, '%s', '&');
%             for n=1:N
%                 if n < N
%                     fprintf(tab, '%s%i%s', '(', n, ') &');
%                 else
%                     fprintf(tab, '%s%i%s\n', '(', n, ') \\');
%                 end
%             end
        end

        function write_row(obj, tab, name_row, values)
            N = length(values);
            fprintf(tab, '%s%s', name_row, '&');
             for n=1:N
                if n < N
                    fprintf(tab, '%2.3f%s', values(n), ' &');
                else
                    fprintf(tab, '%2.3f%s\n', values(n), ' \\');
                end
            end
        end

        function new_row(obj, tab)
              fprintf(tab, '%s', '\n');
        end
        
        function write_hline(obj, tab)
             fprintf(tab, '%s\n', '\hline');
        end

        function panel(obl, tab, num, N)
           fprintf(tab, '%s\n', '\hline\hline');
           fprintf(tab, '%s%i%s%s%s\n', '\multicolumn{', N+1, '}{c}{Panel ', num,'} \\\hline\hline');
        end
       
        function close_table_env(obj, tab)
            fprintf(tab, '%s\n','\end{table}');
        end
        function close_tabular_env(obj, tab)
            fprintf(tab, '%s\n','\end{tabular}');
        end
    end
end