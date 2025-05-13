function plotFunction(paraPairResult, paraReal, numBD,numUPLIM,SNR)
    close all;
    f = figure(1);pos = zeros(numBD,2);
    l=1;
    for i = 1:numUPLIM
        if ~isKey(paraReal, ['BD',int2str(i)])
            continue
        end
        pos(l,:) = paraReal(['BD',int2str(i)]);
        l=l+1;
    end
    plot(pos(:,1),pos(:,2),'bx');
    hold on
    l=1;
    for i = 1:numUPLIM
        if ~isKey(paraReal, ['BD',int2str(i)])
            continue
        end
            annotation(f,'textbox',[0.13-0.05+0.775*(pos(l,1)+90)/180,...
            0.11+0.01+0.815*(pos(l,2)+90)/180 0.1 0.04],...
            'VerticalAlignment','middle',...
            'String',{['Real BD',int2str(i)]},...
            'Interpreter','latex',...
            'HorizontalAlignment','center',...
            'FontSize',12,...
            'FitBoxToText','off',...
            'EdgeColor','none');
            l=l+1;
    end
    l=1;
    for i = 1:numUPLIM
        if ~isKey(paraPairResult, ['BD',int2str(i)])
            continue
        end
        pos(l,:) = paraPairResult(['BD',int2str(i)]);
        l=l+1;
    end

    plot(pos(:,1),pos(:,2),'ro');
    hold on
    l=1;
    for i = 1:numUPLIM
        if ~isKey(paraPairResult, ['BD',int2str(i)])
            continue
        end
        annotation(f,'textbox',[0.13-0.05+0.775*(pos(l,1)+90)/180,...
            0.11-0.06+0.815*(pos(l,2)+90)/180 0.1 0.04],...
            'VerticalAlignment','middle',...
            'String',{['Est BD',int2str(i)]},...
            'Interpreter','latex',...
            'HorizontalAlignment','center',...
            'FontSize',12,...
            'FitBoxToText','off',...
            'EdgeColor','none');
        l=l+1;
    end
    grid on
    axis([-90,90,-90,90])
    xlabel('Estimates of $\theta$ for Receiver 1/${}^\circ$','Interpreter','latex')
    ylabel('Estimates of $\theta$ for Receiver 2/${}^\circ$','Interpreter','latex')
    title(['Target Identification and Localization result at ${\rm SNR}=',int2str(SNR),'$dB'],'Interpreter','latex')
    legend('real parameters of BD','estimates of BD')
end

