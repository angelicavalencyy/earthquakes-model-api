"use strict";(self.webpackChunk_mlflow_mlflow=self.webpackChunk_mlflow_mlflow||[]).push([[3702],{33702:function(e,n,t){t.d(n,{r:function(){return ne}});var a=t(68248),i=t(76118),o=t(27757),r=t(82716),s=t(78171),l=t(11522),d=t(40724),u=t(67063),c=t(66916),m=t(46709),p=t(49620),g=t(42747),h=t(60284),v=t(21723),f=t(73408);var y={name:"a41n9l",styles:"justify-content:flex-start !important"},x={name:"0",styles:""},T={name:"bcffy2",styles:"display:flex;align-items:center;justify-content:space-between"},I={name:"fhxb3m",styles:"display:flex;flex-direction:row;align-items:center"},Y={name:"a41n9l",styles:"justify-content:flex-start !important"};const b=({experimentId:e,currentRunUuid:n,setCompareToRunUuid:t,compareToRunUuid:i,setCurrentRunUuid:s})=>{const{theme:l}=(0,o.u)(),d=(0,g.tz)(),b=(0,p.LE)(),{runInfos:C}=(0,v.Xz)(e),w=(0,m.useMemo)(()=>{if(C)return C.map(e=>{var n;return{key:e.runUuid,value:null!==(n=e.runName)&&void 0!==n?n:e.runUuid}}).filter(e=>e.key)},[C]),R=(0,m.useMemo)(()=>{if(C)return C.filter(e=>e.runUuid!==n).map(e=>{var n;return{key:e.runUuid,value:null!==(n=e.runName)&&void 0!==n?n:e.runUuid}}).filter(e=>Boolean(e.key))},[C,n]),S=null===C||void 0===C?void 0:C.find(e=>e.runUuid===n),U=null===C||void 0===C?void 0:C.find(e=>e.runUuid===i),_=(0,m.useCallback)(n=>{const t=h.Ay.getRunPageRoute(e,n)+"/evaluations",a=new URLSearchParams(window.location.search),i=new URL(t,window.location.origin);a.forEach((e,n)=>{i.searchParams.set(n,e)}),window.location.href=i.toString()},[e]),N=null!==s&&void 0!==s?s:_;return n?(0,f.FD)("div",{css:(0,a.AH)({display:"flex",gap:l.spacing.sm,alignItems:"center"},""),children:[(0,f.Y)("div",{css:(0,a.AH)({display:"flex",alignItems:"center",justifyContent:"flex-start",gap:l.spacing.sm},""),children:(0,f.FD)(r.DialogCombobox,{componentId:"codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_evaluations_evaluationruncompareselector_112",id:"compare-to-run-combobox",value:n?[n]:void 0,children:[(0,f.Y)(r.DialogComboboxCustomButtonTriggerWrapper,{children:(0,f.Y)(o.B,{endIcon:(0,f.Y)(r.ChevronDownIcon,{}),componentId:"mlflow.evaluations_review.table_ui.compare_to_run_button",css:y,children:(0,f.FD)("div",{css:(0,a.AH)({display:"flex",gap:l.spacing.sm,alignItems:"center",fontSize:`${l.typography.fontSizeSm}px !important`},""),children:[(0,f.Y)(u.E,{color:b(n)}),null!==S&&void 0!==S&&S.runName?(0,f.Y)(o.T.Hint,{children:null===S||void 0===S?void 0:S.runName}):d.formatMessage({id:"PUQxu5",defaultMessage:"Select baseline run"})]})})}),(0,f.Y)(r.DialogComboboxContent,{children:(0,f.Y)(r.DialogComboboxOptionList,{children:(w||[]).map((e,t)=>(0,f.Y)(r.DialogComboboxOptionListSelectItem,{value:e.value,onChange:n=>N(e.key),checked:e.key===n,children:(0,f.FD)("div",{css:(0,a.AH)({display:"flex",gap:l.spacing.sm,alignItems:"center"},""),children:[(0,f.Y)(u.E,{color:b(e.key)}),e.value]})},t))})})]})}),(0,f.Y)("span",{css:x,children:d.formatMessage({id:"iYmFCZ",defaultMessage:"compare to"})}),t&&(0,f.Y)("div",{css:T,children:(0,f.FD)("div",{css:I,children:[(0,f.FD)(r.DialogCombobox,{componentId:"codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_evaluations_evaluationruncompareselector_190",id:"compare-to-run-combobox",value:i?[i]:void 0,children:[(0,f.Y)(r.DialogComboboxCustomButtonTriggerWrapper,{children:(0,f.Y)(o.B,{endIcon:(0,f.Y)(r.ChevronDownIcon,{}),componentId:"mlflow.evaluations_review.table_ui.compare_to_run_button",css:Y,children:(0,f.Y)("div",{css:(0,a.AH)({display:"flex",gap:l.spacing.sm,alignItems:"center",fontSize:`${l.typography.fontSizeSm}px !important`},""),children:null!==U&&void 0!==U&&U.runName?(0,f.FD)(f.FK,{children:[(0,f.Y)(u.E,{color:b(i)}),(0,f.Y)(o.T.Hint,{children:null===U||void 0===U?void 0:U.runName})]}):(0,f.Y)("span",{css:(0,a.AH)({color:l.colors.textPlaceholder},""),children:d.formatMessage({id:"XkpMf+",defaultMessage:"baseline run"})})})})}),(0,f.Y)(r.DialogComboboxContent,{children:(0,f.Y)(r.DialogComboboxOptionList,{children:(R||[]).map((e,n)=>(0,f.Y)(r.DialogComboboxOptionListSelectItem,{value:e.value,onChange:n=>t(e.key),checked:e.key===i,children:(0,f.FD)("div",{css:(0,a.AH)({display:"flex",gap:l.spacing.sm,alignItems:"center"},""),children:[(0,f.Y)(u.E,{color:b(e.key)}),e.value]})},n))})})]}),(null===U||void 0===U?void 0:U.runName)&&(0,f.Y)(c.X,{"aria-hidden":"false",css:(0,a.AH)({color:l.colors.textPlaceholder,fontSize:l.typography.fontSizeSm,marginLeft:l.spacing.sm,":hover":{color:l.colors.actionTertiaryTextHover}},""),role:"button",onClick:()=>{t(void 0)},onPointerDownCapture:e=>{e.stopPropagation()}})]})})]}):(0,f.Y)(f.FK,{})};var C=t(31655),w=t(69986),R=t(27763),S=t(54871),U=t(26765);const _=e=>(0,m.useMemo)(()=>e?(0,i.intersection)((0,U.T)(e),[v.o8.Evaluations,v.o8.Metrics,v.o8.Assessments]):[],[e]);var N=t(33656),A=t(88525),D=t(56530),k=t(5690),E=t(38232),F=t(65765),M=t(43233);const H="_assessments.json",L=()=>{const e=(0,D.wA)(),[n,t]=(0,m.useState)(!1);return{savePendingAssessments:(0,m.useCallback)(async(n,a,o)=>{try{t(!0);const r=await(async e=>{const n=(0,F.To)(H,e),t=await(0,F.Up)(n).then(e=>JSON.parse(e));if(!(0,i.isArray)(t.data)||!(0,i.isArray)(t.columns))throw new Error("Artifact is malformed and/or not valid JSON");return t})(n),s=((e,n)=>n.map(n=>{var t,a,i;return[e,n.name,{source_type:null===(t=n.source)||void 0===t?void 0:t.sourceType,source_id:null===(a=n.source)||void 0===a?void 0:a.sourceId,source_metadata:null===(i=n.source)||void 0===i?void 0:i.metadata},n.timestamp||null,n.booleanValue||null,n.numericValue||null,n.stringValue||null,n.rationale||null,n.metadata||null,null,null]}))(a,o),l=((e,n,t)=>{const a=(0,E.G4)(H,n),o=t.map(({name:e,source:n})=>({name:e,source:n?{source_type:n.sourceType,source_id:n.sourceId,source_metadata:n.metadata}:void 0})),r=a.entries.filter(({evaluation_id:n,name:t,source:a})=>e===n&&o.find(e=>(0,i.isEqual)({name:t,source:a},e))).map(e=>a.entries.indexOf(e));return n.data.filter((e,n)=>!r.includes(n))})(a,r,o);await e((0,k.Of)(n,H,{columns:r.columns,data:[...l,...s]})),e({type:(0,M.ec)(k.So),payload:(0,E.G4)(H,{columns:r.columns,data:[...l,...s]}),meta:{runUuid:n,artifactPath:H}})}catch(r){throw l.A.logErrorAndNotifyUser(r.message||r),r}finally{t(!1)}},[e]),isSaving:n}};var $=t(81641);const O=$.J1`
  query SearchRuns($data: MlflowSearchRunsInput!) {
    mlflowSearchRuns(input: $data) {
      apiError {
        helpUrl
        code
        message
      }
      runs {
        info {
          runName
          status
          runUuid
          experimentId
          artifactUri
          endTime
          lifecycleStage
          startTime
          userId
        }
        experiment {
          experimentId
          name
          tags {
            key
            value
          }
          artifactLocation
          lifecycleStage
          lastUpdateTime
        }
        data {
          metrics {
            key
            value
            step
            timestamp
          }
          params {
            key
            value
          }
          tags {
            key
            value
          }
        }
        inputs {
          datasetInputs {
            dataset {
              digest
              name
              profile
              schema
              source
              sourceType
            }
            tags {
              key
              value
            }
          }
          modelInputs {
            modelId
          }
        }
        outputs {
          modelOutputs {
            modelId
            step
          }
        }
        modelVersions {
          version
          name
          creationTimestamp
          status
          source
        }
      }
    }
  }
`,P=({filter:e,experimentIds:n,disabled:t=!1})=>{var a,o,r;const{data:s,loading:l,error:d,refetch:u}=(0,$.IT)(O,{variables:{data:{filter:e,experimentIds:n}},skip:t});return{loading:l,data:(0,i.first)(null!==(a=null===s||void 0===s||null===(o=s.mlflowSearchRuns)||void 0===o?void 0:o.runs)&&void 0!==a?a:[]),refetchRun:u,apolloError:d,apiError:null===s||void 0===s||null===(r=s.mlflowSearchRuns)||void 0===r?void 0:r.apiError}};var B={name:"r3950p",styles:"flex:1;display:flex;justify-content:center;align-items:center"};const j=({experimentId:e,runUuid:n,runTags:t,runDisplayName:i,data:l})=>{const{theme:u}=(0,o.u)(),c=_(t),m=0===(null===l||void 0===l?void 0:l.length),[p,g]=(0,s.Y)(),h=(0,N.N9)(),y=L(),{data:x,displayName:T,loading:I}=z(e,p,c);if(I)return(0,f.Y)(r.LegacySkeleton,{});const Y=e=>e.filter(e=>e.type===v.$6.ASSESSMENT||e.type===v.$6.INPUT||e.type===v.$6.TRACE_INFO&&[v.tj,v.$W,v.Pn].includes(e.id));return m?(0,f.Y)("div",{css:B,children:(0,f.Y)(r.Empty,{title:(0,f.Y)(d.A,{id:"NqqMPs",defaultMessage:"No evaluation tables logged"}),description:null})}):(0,f.FD)("div",{css:(0,a.AH)({marginTop:u.spacing.sm,width:"100%",overflowY:"hidden"},""),children:[(0,f.Y)("div",{css:(0,a.AH)({width:"100%",padding:`${u.spacing.xs}px 0`},""),children:(0,f.Y)(b,{experimentId:e,currentRunUuid:n,compareToRunUuid:p,setCompareToRunUuid:g})}),(()=>{const t={experimentId:e,currentRunDisplayName:i,currentEvaluationResults:l||[],compareToEvaluationResults:x,runUuid:n,compareToRunUuid:p,compareToRunDisplayName:T,compareToRunLoading:I,saveAssessmentsQuery:y,getTrace:w.Rb,initialSelectedColumns:Y};return(0,f.Y)(v.tU,{makeHtml:h,children:(0,f.Y)(v.js,{...t})})})()]})},z=(e,n,t)=>{const{data:a,isLoading:o}=(0,v.Ie)({runUuid:n||"",artifacts:t},{disabled:(0,i.isNil)(n)}),{data:r,loading:s}=P({experimentIds:[e],filter:`attributes.run_id = "${n}"`,disabled:(0,i.isNil)(n)});return{data:a,displayName:l.A.getRunDisplayName(null===r||void 0===r?void 0:r.info,n),loading:o||s}};var Q=t(39595),q=t(82636),V=t(58663),G=t(27462),K=t(7655),W=t(55532),X=t(25862);const J=({children:e,makeHtmlFromMarkdown:n,experimentId:t})=>(0,f.Y)(v.tU,{makeHtml:n,children:e});var Z={name:"1nxh63r",styles:"overflow-y:hidden;height:100%;display:flex;flex-direction:column"};const ee=({experimentId:e,runUuid:n,runDisplayName:t,setCurrentRunUuid:i,showCompareSelector:r=!1,showRefreshButton:l=!1})=>{const{theme:c}=(0,o.u)(),g=((0,K.A)(),(0,N.N9)()),[h,y]=(0,s.Y)(),x=(0,m.useMemo)(()=>[(0,v.$U)(e)],[e]),T=w.Uv,I=!1,{assessmentInfos:Y,allColumns:U,totalCount:_,evaluatedTraces:D,otherEvaluatedTraces:k,isLoading:E,error:F,tableFilterOptions:M}=(0,v.KW)({locations:x,runUuid:n,otherRunUuid:h,disabled:I,filterByAssessmentSourceRun:!0}),[H,L]=(0,m.useState)({});(0,S.Gt)("selectedTraceIds",H);const[$,O]=(0,m.useState)(""),[P,B]=(0,v.R7)(),j=(0,p.LE)(),z=(0,Q.jE)(),ee=(0,m.useCallback)(e=>{const{responseHasContent:n,inputHasContent:t,tokensHasContent:a}=(0,q.l)(D.concat(k));return e.filter(e=>e.type===v.$6.ASSESSMENT||e.type===v.$6.EXPECTATION||t&&e.type===v.$6.INPUT||n&&e.type===v.$6.TRACE_INFO&&e.id===v.Rl||a&&e.type===v.$6.TRACE_INFO&&e.id===v.YO||e.type===v.$6.TRACE_INFO&&[v.XQ,v.tj,v.$W].includes(e.id))},[D,k]),{selectedColumns:ne,toggleColumns:ie,setSelectedColumns:oe}=(0,v.K0)(e,U,ee,n),[re,se]=(0,v.GY)(ne),{data:le,isLoading:de,isFetching:ue,error:ce,refetchMlflowTraces:me}=(0,v.Zn)({locations:x,currentRunDisplayName:t,searchQuery:$,filters:P,runUuid:n,tableSort:re,disabled:I,filterByAssessmentSourceRun:!0}),{data:pe,displayName:ge,loading:he}=ae({experimentId:e,traceLocations:x,compareToRunUuid:h,isQueryDisabled:I}),ve=(0,m.useMemo)(()=>({currentCount:null===le||void 0===le?void 0:le.length,logCountLoading:de,totalCount:_,maxAllowedCount:(0,C.pR)()}),[le,de,_]),{showEditTagsModalForTrace:fe,EditTagsModal:ye}=(0,A.$)({onSuccess:()=>(0,v.BL)({queryClient:z}),existingTagKeys:(0,v.d9)(le||[])}),xe=(0,G.F)({traceSearchLocations:x}),Te=W.p,Ie=(0,m.useMemo)(()=>({deleteTracesAction:xe,exportToEvals:!0,editTags:{showEditTagsModalForTrace:fe,EditTagsModal:ye}}),[xe,fe,ye]),Ye=de||he,be=j(n),Ce=h?j(h):void 0,[we,Re]=(0,m.useState)(!1);return E?(0,f.Y)(te,{}):F?(0,f.Y)("div",{children:(0,f.Y)("pre",{children:String(F)})}):(0,f.FD)("div",{css:(0,a.AH)({marginTop:c.spacing.sm,width:"100%",overflowY:"hidden"},""),children:[!r&&(0,f.Y)("div",{css:(0,a.AH)({width:"100%",padding:`${c.spacing.xs}px 0`},""),children:(0,f.Y)(b,{experimentId:e,currentRunUuid:n,compareToRunUuid:h,setCompareToRunUuid:y,setCurrentRunUuid:i})}),r&&h&&(0,f.FD)("div",{css:(0,a.AH)({display:"flex",alignItems:"center",width:"100%",paddingBottom:c.spacing.sm,gap:c.spacing.sm},""),children:[(0,f.Y)(o.T.Text,{children:(0,f.Y)(d.A,{id:"v/njcq",defaultMessage:"Comparing"})}),(0,f.FD)("span",{css:(0,a.AH)({display:"inline-flex",alignItems:"center",gap:c.spacing.xs},""),children:[be&&(0,f.Y)(u.E,{color:be}),(0,f.Y)(o.T.Text,{bold:!0,children:t})]}),(0,f.Y)(o.T.Text,{children:(0,f.Y)(d.A,{id:"taI4Bv",defaultMessage:"to"})}),(0,f.FD)("span",{css:(0,a.AH)({display:"inline-flex",alignItems:"center",gap:c.spacing.xs},""),children:[Ce&&(0,f.Y)(u.E,{color:Ce}),(0,f.Y)(o.T.Text,{bold:!0,children:ge})]})]}),(0,f.Y)(V.gm,{renderExportTracesToDatasetsModal:Te,DrawerComponent:X.q,children:(0,f.Y)(R.Nf,{rowSelection:H,setRowSelection:L,children:(0,f.Y)(v.sG,{experimentId:e,children:(0,f.FD)("div",{css:Z,children:[(0,f.Y)(v.w_,{experimentId:e,searchQuery:$,setSearchQuery:O,filters:P,setFilters:B,assessmentInfos:Y,countInfo:ve,traceActions:Ie,tableSort:re,setTableSort:se,allColumns:U,selectedColumns:ne,setSelectedColumns:oe,toggleColumns:ie,traceInfos:le,tableFilterOptions:M,onRefresh:l?me:void 0,isRefreshing:l?ue:void 0}),Ye?(0,f.Y)(te,{}):ce?(0,f.Y)("div",{children:(0,f.Y)("pre",{children:String(ce)})}):(0,f.Y)(J,{makeHtmlFromMarkdown:g,experimentId:e,children:(0,f.Y)(v._p,{experimentId:e,currentRunDisplayName:t,compareToRunDisplayName:ge,compareToRunUuid:h,getTrace:T,getRunColor:j,assessmentInfos:Y,setFilters:B,filters:P,selectedColumns:ne,allColumns:U,tableSort:re,currentTraceInfoV3:le||[],compareToTraceInfoV3:pe,onTraceTagsEdit:fe,displayLoadingOverlay:!1})}),ye]})})})})]})},ne=({experimentId:e,experiment:n,runUuid:t,runTags:a,runDisplayName:o,setCurrentRunUuid:r,showCompareSelector:s=!1,showRefreshButton:l=!1})=>{const d=_(a),u=Boolean(t),{data:c,isLoading:m}=(0,v.Ie)({runUuid:t||"",artifacts:d||void 0},{disabled:!u});return m?(0,f.Y)(te,{}):!(0,i.isNil)(c)&&c.length>0?(0,f.Y)(j,{experimentId:e,runUuid:t,runDisplayName:o,data:c,runTags:a}):(0,f.Y)(ee,{experimentId:e,runUuid:t,runDisplayName:o,setCurrentRunUuid:r,showCompareSelector:s,showRefreshButton:l})},te=()=>{const{theme:e}=(0,o.u)();return(0,f.Y)("div",{css:(0,a.AH)({display:"block",marginTop:e.spacing.md,height:"100%",width:"100%"},""),children:[...Array(10).keys()].map(e=>(0,f.Y)(r.ParagraphSkeleton,{label:"Loading...",seed:`s-${e}`},e))})},ae=e=>{const{compareToRunUuid:n,experimentId:t,traceLocations:a,isQueryDisabled:o}=e,{data:r,isLoading:s}=(0,v.Zn)({locations:a,currentRunDisplayName:void 0,runUuid:n,disabled:(0,i.isNil)(n)||o,filterByAssessmentSourceRun:!0}),{data:d,loading:u}=P({experimentIds:[t],filter:`attributes.run_id = "${n}"`,disabled:(0,i.isNil)(n)});return{data:r,displayName:l.A.getRunDisplayName(null===d||void 0===d?void 0:d.info,n),loading:s||u}}},78171:function(e,n,t){t.d(n,{Y:function(){return r}});var a=t(46709),i=t(91105);const o="compareToRunUuid",r=()=>{var e;const[n,t]=(0,i.ok)();return[null!==(e=n.get(o))&&void 0!==e?e:void 0,(0,a.useCallback)(e=>{t(n=>void 0===e?(n.delete(o),n):(n.set(o,e),n))},[t])]}}}]);