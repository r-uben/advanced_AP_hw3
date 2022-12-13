classdef modFileGenerator
   methods

       function calibration(obj, fid)
             fprintf(fid, '// CALIBRATION\n\n');
       end
  
       %% VARIABLES
       function variables(obj, fid, vars, exovars)
           strvars = "";
           % Print the endogenous variables
           for i=1:length(vars)
               strvars = strjoin([strvars vars(i)]);
           end
           fprintf(fid, ...
               strjoin(["var" " " strjoin(strvars), ";\n"]));
           % Print the exogenous variabels
           strexovars = "";
           for i=1:length(exovars)
               strexovars = strjoin([strexovars exovars(i)]);
           end
           fprintf(fid, ...
               strjoin(["varexo" " " strjoin(strexovars) ";\n\n"]));
       end
       %% PARAMETERS
       function parameters(obj, fid, params)
            param_names = params.keys();
            strparams = "";
            for i=1:length(param_names)
                strparams = strjoin([strparams param_names(i)]);
            end
            fprintf(fid, ...
                strjoin(["parameters ", strjoin(strparams), ";\n\n"]));
            for i=1:length(param_names)
                param = replace(param_names(i), " ", "");
                fprintf(fid, strjoin([param "= %e; \n"]), params(param));
                if i==length(param_names)
                    fprintf(fid,"\n");
                end
            end
       end

       %% MODEL
       function model(obj, fid, equations)
            vars = equations.keys();
            for i=1:length(vars)
                var = vars(i);
                eq = replace(equations(var), " ", "");
                fprintf(fid, strjoin([eq "\n"]));
                if i==length(vars)
                    fprintf(fid,"\n");
                end
            end
       end

       %% BONDS & ZCE

       function security(obj, fid, p, r, exr)
           newline = ";\n";
           p_keys = p.keys();
           r_keys = r.keys();
           exr_keys = exr.keys();
            for n=1:length(p_keys)
                try
                    fprintf(fid, strjoin(p(p_keys(n)), newline));
                    fprintf(fid, strjoin(r(r_keys(n)), newline));
                    fprintf(fid, strjoin(exr(exr_keys(n)), newline));
                catch
                    continue
                end
                if n==length(p_keys)
                fprintf(fid,"\n");
                end
            end
       end


       %% INITIAL VALUES
       function initial_values(obj, fid, init_values)
           initval_names = init_values.keys();
            for i=1:length(initval_names)
                initval = replace(initval_names(i), " ", "");
                fprintf(fid, strjoin([initval "= %e; \n"]), init_values(initval));
                if i==length(initval_names)
                    fprintf(fid,"\n");
                end
            end
       end

       function sec_initial_values(obj, fid, p, r, exr);
           newline = ";\n";
           p_keys = p.keys();
           r_keys = r.keys();
           exr_keys = exr.keys();
           for n=1:length(p_keys)
               try
                   fprintf(fid, strjoin(p(p_keys(n)),newline));
                   fprintf(fid,strjoin(r(r_keys(n)),newline));
                   fprintf(fid,strjoin(exr(exr_keys(n)),newline));
               catch
                   continue
               end

               if n==length(p_keys)
                    fprintf(fid,"\n");
               end
           end
       end

   end
end