document.onload = (function(d3, saveAs, Blob, undefined){
    "use strict";

    var settings = {
      appendElSpec: "#graph"
    };
    // define graphcreator object
    var GraphCreator = function(svg, nodes, edges){
        var thisGraph = this;
            thisGraph.idct = 0;

        thisGraph.nodes = nodes || [];
        thisGraph.edges = edges || [];
    
        thisGraph.state = {
            selectedNode: null,
            selectedEdge: null,
            mouseDownNode: null,
            justDragged: false,
            justScaleTransGraph: false
        };

        thisGraph.svg = svg;
        thisGraph.svgG = svg.append("g")
                .classed(thisGraph.consts.graphClass, true);
        var svgG = thisGraph.svgG;
    
        // svg nodes and edges
        thisGraph.paths = svgG.append("g").selectAll("g");
        thisGraph.circles = svgG.append("g").selectAll("g");
  
        thisGraph.drag = d3.behavior.drag()
                .origin(function(d){
                return {x: d.x, y: d.y};
                })
                .on("drag", function(args){
                thisGraph.state.justDragged = true;
                thisGraph.dragmove.call(thisGraph, args);
                })
  
       // listen for key events
        svg.on("mousedown", function(d){thisGraph.svgMouseDown.call(thisGraph, d);});
        svg.on("mouseup", function(d){thisGraph.svgMouseUp.call(thisGraph, d);});
    
        // listen for dragging
        var dragSvg = d3.behavior.zoom()
                .on("zoom", function(){
                    thisGraph.zoomed.call(thisGraph);
                    return true;
                })
                .on("zoomstart", function(){
                    var ael = d3.select("#" + thisGraph.consts.activeEditId).node();
                    if (ael){
                        ael.blur();
                    }
                    if (!d3.event.sourceEvent.shiftKey) d3.select('body').style("cursor", "move");
                })
                .on("zoomend", function(){
                    d3.select('body').style("cursor", "auto");
                });
    
        svg.call(dragSvg).on("dblclick.zoom", null);
  
       // listen for resize
        window.onresize = function(){thisGraph.updateWindow(svg);};
    
        // handle download data
        d3.select("#download-input").on("click", function(){
            var saveEdges = [];
            thisGraph.edges.forEach(function(val, i){
            saveEdges.push({source: val.source.id, target: val.target.id});
            });
            var blob = new Blob([window.JSON.stringify({"nodes": thisGraph.nodes, "edges": saveEdges})], {type: "text/plain;charset=utf-8"});
            saveAs(blob, "mySimulation.json");
        });
    };
  
    GraphCreator.prototype.setIdCt = function(idct){
        this.idct = idct;
    };
  
    GraphCreator.prototype.consts =  {
        selectedClass: "selected",
        circleGClass: "conceptG",
        PhysicsModule: "physics-module",
        ComputeTool: "compute-tool",
        Diagnostic: "diagonstic",
        graphClass: "graph",
        activeEditId: "active-editing",
        BACKSPACE_KEY: 8,
        DELETE_KEY: 46,
        ENTER_KEY: 13,
        nodeHeight: 25, 
        rectScale: 10
    };
  
    /* PROTOTYPE FUNCTIONS */
    GraphCreator.prototype.dragmove = function(d) {
        var thisGraph = this;
        d.x += d3.event.dx;
        d.y +=  d3.event.dy;
        thisGraph.updateGraph();
    };
  
    /* insert svg line breaks: taken from http://stackoverflow.com/questions/13241475/how-do-i-include-newlines-in-labels-in-d3-charts */
    GraphCreator.prototype.insertTitleLinebreaks = function (gEl, title) {
        var words = title.split(/\s+/g)
        gEl.selectAll("text").remove();
        var el = gEl.append("text")
                .attr("text-anchor","middle")
                .attr("dy", "2")
                .style("font-family", "andale mono")
                .style("font-size", "10")
                .style("font-weight", "300");
    
        for (var i = 0; i < words.length; i++) {
            var tspan = el.append('tspan').text(words[i]);
            if (i > 0)
            tspan.attr('x', 0).attr('dy', '15');
        }
    };
  
    // mousedown on node
    GraphCreator.prototype.circleMouseDown = function(d3node, d){
        var thisGraph = this,
            state = thisGraph.state;
        d3.event.stopPropagation();
        state.mouseDownNode = d;
    };

    // mouseup on nodes
    GraphCreator.prototype.circleMouseUp = function(d3node, d){
      var thisGraph = this,
          state = thisGraph.state,
          consts = thisGraph.consts;
  
      var mouseDownNode = state.mouseDownNode;
  
      if (!mouseDownNode) return;
  
      thisGraph.dragLine.classed("hidden", true);
  
      if (mouseDownNode !== d){
        // we're in a different node: create new edge for mousedown edge and add to graph
        var newEdge = {source: mouseDownNode, target: d};
        var filtRes = thisGraph.paths.filter(function(d){
          if (d.source === newEdge.target && d.target === newEdge.source){
            thisGraph.edges.splice(thisGraph.edges.indexOf(d), 1);
          }
          return d.source === newEdge.source && d.target === newEdge.target;
        });
        if (!filtRes[0].length){
          thisGraph.edges.push(newEdge);
          thisGraph.updateGraph();
        }
      } else{
        // we're in the same node
        if (state.justDragged) {
          // dragged, not clicked
          state.justDragged = false;
        }
      }
      state.mouseDownNode = null;
      return;
  
    };
  
    // mousedown on main svg
    GraphCreator.prototype.svgMouseDown = function(){
      this.state.graphMouseDown = true;
    };
  
    // mouseup on main svg
    GraphCreator.prototype.svgMouseUp = function(){
        var thisGraph = this,
            state = thisGraph.state;
        if (state.justScaleTransGraph) {
            // dragged not clicked
            state.justScaleTransGraph = false;
        } 
        state.graphMouseDown = false;
    };

    // call to propagate changes to graph
    GraphCreator.prototype.updateGraph = function(){
        var thisGraph = this,
            consts = thisGraph.consts,
            state = thisGraph.state;
    
        thisGraph.paths = thisGraph.paths.data(thisGraph.edges, function(d){
            return String(d.source.id) + "+" + String(d.target.id);
        });
        var paths = thisGraph.paths;
        // update existing paths
        paths.style('marker-end', 'url(#end-arrow)')
            .classed(consts.selectedClass, function(d){
            return d === state.selectedEdge;
            })
            .attr("d", function(d){
            return "M" + d.source.x + "," + d.source.y + "L" + d.target.x + "," + d.target.y;
            });
    
        // add new paths
        paths.enter()
            .append("path")
            .style('stroke-width', '1px')
            .classed("link", true)
            .attr("d", function(d){
                return "M" + d.source.x + "," + d.source.y + "L" + d.target.x + "," + d.target.y;
            })
            .on("mousedown", function(d){
                thisGraph.pathMouseDown.call(thisGraph, d3.select(this), d);
            }
            )
            .on("mouseup", function(d){
            });
    
        // remove old links
        paths.exit().remove();
    
        // update existing nodes
        thisGraph.circles = thisGraph.circles.data(thisGraph.nodes, function(d){ return d.id;});
        thisGraph.circles.attr("transform", function(d){return "translate(" + d.x + "," + d.y + ")";});
    
        // add new nodes
        var newGs= thisGraph.circles.enter()
                .append("g");
                
    
        newGs.classed(consts.circleGClass, true)
            .attr("transform", function(d){return "translate(" + d.x + "," + d.y + ")";})
            .on("mouseover", function(d){
                switch(d.type) {
                    case 'Diagnostic':
                        d3.select(this).classed(consts.Diagnostic, true);
                        break;
                    case 'PhysicsModule':
                        d3.select(this).classed(consts.PhysicsModule, true);
                        break;
                    case 'ComputeTool':
                        d3.select(this).classed(consts.ComputeTool, true);
                        break;
                }
                thisGraph.insertTitleLinebreaks(d3.select(this), d.type);
            })
            .on("mouseout", function(d){
                thisGraph.insertTitleLinebreaks(d3.select(this), d.title);
            })
            .on("mousedown", function(d){
                thisGraph.insertTitleLinebreaks(d3.select(this), d.title);
                d3.select(this).classed(consts.circleGClass, true);
                thisGraph.circleMouseDown.call(thisGraph, d3.select(this), d);
            })
            .on("mouseup", function(d){
                thisGraph.insertTitleLinebreaks(d3.select(this), d.type);
                thisGraph.circleMouseUp.call(thisGraph, d3.select(this), d);
            })
            .call(thisGraph.drag);
    
        newGs.append("rect")
        .attr("rx", 3)
        .attr("ry", 3)
        .attr("width", function(d) { return d.title.length * consts.rectScale; })
        .attr("height", consts.nodeHeight)
        .attr("x", function(d) { return -1 * d.title.length * consts.rectScale/ 2; })
        .attr("y", -consts.nodeHeight/2)
        .style("stroke", "black")
        .style("stroke-width", "1px");
    
        newGs.each(function(d){
            thisGraph.insertTitleLinebreaks(d3.select(this), d.title);
        });
        // remove old nodes
        thisGraph.circles.exit().remove();
    };
  
    GraphCreator.prototype.zoomed = function(){
      this.state.justScaleTransGraph = true;
      d3.select("." + this.consts.graphClass)
        .attr("transform", "translate(" + d3.event.translate + ") scale(" + d3.event.scale + ")");
    };
  
    GraphCreator.prototype.updateWindow = function(svg){
      var docEl = document.documentElement,
          bodyEl = document.getElementsByTagName('body')[0];
      var x = window.innerWidth || docEl.clientWidth || bodyEl.clientWidth;
      var y = window.innerHeight|| docEl.clientHeight|| bodyEl.clientHeight;
      svg.attr("width", x).attr("height", y);
    };

    /**** MAIN ****/
    var docEl = document.documentElement,
        bodyEl = document.getElementsByTagName('body')[0];
  
    var width = window.innerWidth || docEl.clientWidth || bodyEl.clientWidth,
        height =  window.innerHeight|| docEl.clientHeight|| bodyEl.clientHeight;
  
    
    var nodes = mydata.nodes;
    var edges = mydata.edges;
    edges.forEach(function(e, i){
        edges[i] = {source: nodes.filter(function(n){return n.id == e.source;})[0],
                    target: nodes.filter(function(n){return n.id == e.target;})[0]};
    });
   
    /** MAIN SVG **/
    var svg = d3.select(settings.appendElSpec).append("svg")
          .attr("width", width)
          .attr("height", height);
    var graph = new GraphCreator(svg, nodes, edges);
        graph.setIdCt(2);
    graph.updateGraph();
  })(window.d3, window.saveAs, window.Blob);