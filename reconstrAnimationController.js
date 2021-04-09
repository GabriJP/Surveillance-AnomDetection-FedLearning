/* File: reconstrAnimationController.js
 * Author: Nicolás Cubero
 * Description: Code Implementation for DemoPlayer class.
 * Controller class for the interactive display of the event detection
 *	progress made by the model for a test sample.
 *	The play shows a live displaying of the
 *	video reconstruction made by a model and the reconstruction error computed
 *	by the model on each frame.
*/
export default class ReconstrAnimationController {

	constructor(elem_obj, orig_rec_img_src, repl_rec_img_src, rect_measures) {

		/* Builds a ReconstrAnimationController instance.
		 * Parameters:
		 * 	- elem_obj: test_demo class HTML div element containing the original
		 *		video sample, the reconstruction made by the model for the
		 *		sample, and the normalized reconstruction error graph.
		 *
		 *
		 *	- orig_rec_image: URL to the reconstruction image graph containing the
		 *		the ground truth.
		 *
		 *	- _repl_rec_image: URL to the reconstruction image graph containing the
		 *		prediction to replace the ground truth graph.
		 *
		 *	- rect_measures: Meausures of the rectangle whitin the frame range of
		 *		the reconstruction error graph. A JSON document containing the
		 *		standarized rectangle corners with the keys x1, x2, y1 and y2
		 *		must be passed.
		 *
		 *	 	Example:
		 *		{
		 *			"x1": 0.095325055,
		 *			"y1": 0.071304348,
		 *			"x2": 0.972242513,
		 *			"y2": 0.880869565
		 *		}
		 *
		*/

		this._timeout = null;
		var that = this;

		// Find the IDs for all the elements
		var x = this._parse(elem_obj);

		this._orig_vid_id = x[0];
		this._rec_vid_id = x[1];
		this._org_rec_ele = x[2];
		this._repl_rec_img_src = repl_rec_img_src;

		this._canvas_rec_img = null;
		this._repl_rec_image = null;
		this._canvas_supported = null;
		this._rect_shape = {...rect_measures};

		// Configure the video play
		this._orig_vid_id.loop = true;
		this._rec_vid_id.loop = true;

		this._orig_vid_id.controls = false;
		this._rec_vid_id.controls = false;

		// Create the original image
		this._org_rec_img = new Image();
		this._org_rec_img.onload = function() {
			that._setup(); // Install the canvas when the image is fully loaded
		}
		this._org_rec_img.src = orig_rec_img_src;

	}

	_parse(elem_obj) {
		return [elem_obj.getElementsByClassName("demo_video original")[0].getElementsByTagName("video")[0],
			elem_obj.getElementsByClassName("demo_video reconstructed")[0].getElementsByTagName("video")[0],
			elem_obj.getElementsByClassName("demo_rec_error")[0]];
	}

	_setup() {

		var that = this;
		var scale = this._org_rec_ele.clientWidth / this._org_rec_img.naturalWidth;

		// Create the Canvas and put the image into
		this._canvas_rec_img = document.createElement('canvas');


		// 	Check browser canvas support
		if (!(this._canvas_rec_img.getContext &&
																			this._canvas_rec_img.getContext('2d'))) {
			this._canvas_supported = false;

			/* Install the original image (ground truth) as static image with no
				 animation, in case that Canvas is not supported*/
			this._org_rec_ele.appendChild(this._org_rec_img);
		}
		else {
			this._canvas_supported = true;


			this._canvas_rec_img.width = scale * that._org_rec_img.naturalWidth;
			this._canvas_rec_img.height = scale * that._org_rec_img.naturalHeight;

			// Compute the image area to be progressively displayed
			that._rect_shape["x1"] *= this._canvas_rec_img.width;
			that._rect_shape["y1"] *= this._canvas_rec_img.height;
			that._rect_shape["x2"] *= this._canvas_rec_img.width;
			that._rect_shape["y2"] *= this._canvas_rec_img.height;

			that._rect_shape["width"] = that._rect_shape["x2"] - that._rect_shape["x1"];
			that._rect_shape["height"] = that._rect_shape["y2"] - that._rect_shape["y1"];

			// Create the replace rec. image
			this._repl_rec_img = new Image();
			this._repl_rec_img.src = this._repl_rec_img_src;

			// Install canvas
			this._org_rec_ele.appendChild(this._canvas_rec_img);
			window.requestAnimationFrame(ev => that._render_rec_img());
		}

	}

	play() {

		/*
		 * Callback function for starting the play of original and reconstruction
		 *  videos and the reconstruction animation
		 */
		if (this._orig_vid_id.error) {
			console.error('Cannot play the video. MediaError Object code: ' +
											this._orig_vid_id.error);
			return;
		}

		this._orig_vid_id.play();
		this._rec_vid_id.play();

		var that = this;

		if (!this._timeout && this._canvas_supported) {
			this._timeout = window.requestAnimationFrame(ev => that._render_rec_img());
		}
	}

	stop() {
		/*
		 * Callback function for stopping the play of original and reconstruction
		 *  videos and the reconstruction animation
		*/
		if (this._orig_vid_id.error) {
			return;
		}

		this._orig_vid_id.pause();
		this._orig_vid_id.currentTime = 0;

		this._rec_vid_id.pause();
		this._rec_vid_id.currentTime = 0;
	}


	_render_rec_img() {

		/* Renders a rec. image animation frame */
		var ctx = this._canvas_rec_img.getContext("2d");

		// Clear animation content
		ctx.clearRect(0, 0, this._canvas_rec_img.width, this._canvas_rec_img.height);

		if (!this._orig_vid_id.paused) {

			/* Draw a new frame */

			// Draw the error rec image
			ctx.drawImage(this._repl_rec_img, 0, 0, this._repl_rec_img.width,
										this._repl_rec_img.height, 0, 0, this._canvas_rec_img.width,
										this._canvas_rec_img.height);

			// Clear the portion of the rec. image to be not yet displayed
			var width = (1 - (this._orig_vid_id.currentTime /
												this._orig_vid_id.duration)) * this._rect_shape["width"];
			ctx.fillStyle = "#FFFFFF";

			ctx.clearRect(this._rect_shape["x2"] - width,
					this._rect_shape["y1"],
					width,
					this._rect_shape["height"]);

			var that = this;

			// Schedule for the next frame if playing is not stopped
			this._timeout = window.requestAnimationFrame(ev => that._render_rec_img());
		}
		else {
			/* Stop animation */

			// Draw the error rec image
			ctx.drawImage(this._org_rec_img, 0, 0, this._org_rec_img.width,
										this._org_rec_img.height, 0, 0, this._canvas_rec_img.width,
										this._canvas_rec_img.height);

			this._timeout = null; // Remove schedule for next frame
		}

	}

}
