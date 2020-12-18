package app.view;

import java.util.Optional;
import java.util.function.Consumer;

import app.GlobeData;
import app.util.Triplet;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

public class WorldSizePicker extends Pane{
	VBox options = new VBox(10);
	Optional<Consumer<Triplet<Integer, Integer, Integer>>> onSelect = Optional.empty();
	int base = 48;
	public WorldSizePicker() {
		addOption("Mini",   new Triplet<>(base  , base*2, 20));
		addOption("Small",  new Triplet<>(base*2, base*4, 50));
		addOption("Medium", new Triplet<>(base*3, base*6, 70));
		addOption("Big",    new Triplet<>(base*4, base*8, 75));
		addOption("Large",  new Triplet<>(base*5, base*10, 75));
		addOption("Huge",   new Triplet<>(base*6, base*12, 100));
		
		
		this.getChildren().add(options);
		options.layoutXProperty().bind(widthProperty().divide(2).subtract(options.widthProperty().divide(2)));
		options.layoutYProperty().bind(heightProperty().divide(2).subtract(options.heightProperty().divide(2)));
	}
	private void addOption(String string, Triplet<Integer, Integer, Integer> t) {
		Button button = new Button(string);
		Tooltip tip = new Tooltip(formatSize(t, GlobeData.byteSizeIf(t.a, t.b, t.c)));
		Tooltip.install(button, tip);
		button.setOnAction(e->onSelect.ifPresent(c->c.accept(t)));
		options.getChildren().add(button);
	}
	private String formatSize(Triplet<Integer, Integer, Integer> t, long b) {
		String s;
		if(b < 1024) {
			s= b+" bytes";
		}else {
			b/=1024;
			if(b < 1024) {
				s= b+" megabytes";
			}else {
				b/=1024;
				if(b < 1024) {
					s= b+" killobytes";
				}else {
					b/=1024;
					s= b+" gigabytes";
				}
			}
		}
		return String.format("%d x %d x %d ( %s )", t.a, t.b, t.c, s);
	}
	
	public void setOnSelect(Consumer<Triplet<Integer, Integer, Integer>> onSelect) {
		this.onSelect = Optional.ofNullable(onSelect);
	}
	
}
